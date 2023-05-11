# SPDX-License-Identifier: LGPL-2.1+

import asyncio
import base64
import contextlib
import hashlib
import logging
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterator, Optional

from mkosi.config import ConfigFeature, MkosiArgs, MkosiConfig
from mkosi.install import copy_path
from mkosi.log import die
from mkosi.run import MkosiAsyncioThread, run, spawn
from mkosi.types import PathString
from mkosi.util import (
    Distribution,
    OutputFormat,
    qemu_check_kvm_support,
    qemu_check_vsock_support,
    tmp_dir,
)


def machine_cid(config: MkosiConfig) -> int:
    cid = int.from_bytes(hashlib.sha256(config.output_with_version.encode()).digest()[:4], byteorder='little')
    # Make sure we don't return any of the well-known CIDs.
    return max(3, min(cid, 0xFFFFFFFF - 1))


def find_qemu_binary(config: MkosiConfig) -> str:
    binaries = ["qemu", "qemu-kvm", f"qemu-system-{config.architecture}"]
    for binary in binaries:
        if shutil.which(binary) is not None:
            return binary

    die("Couldn't find QEMU/KVM binary")


def find_qemu_firmware(config: MkosiConfig) -> tuple[Path, bool]:
    FIRMWARE_LOCATIONS = {
        "x86_64": ["/usr/share/ovmf/x64/OVMF_CODE.secboot.fd"],
        "i386": [
            "/usr/share/edk2/ovmf-ia32/OVMF_CODE.secboot.fd",
            "/usr/share/OVMF/OVMF32_CODE_4M.secboot.fd"
        ],
    }.get(config.architecture, [])

    for firmware in FIRMWARE_LOCATIONS:
        if os.path.exists(firmware):
            return Path(firmware), True

    FIRMWARE_LOCATIONS = {
        "x86_64": [
            "/usr/share/ovmf/ovmf_code_x64.bin",
            "/usr/share/ovmf/x64/OVMF_CODE.fd",
            "/usr/share/qemu/ovmf-x86_64.bin",
        ],
        "i386": ["/usr/share/ovmf/ovmf_code_ia32.bin", "/usr/share/edk2/ovmf-ia32/OVMF_CODE.fd"],
        "aarch64": ["/usr/share/AAVMF/AAVMF_CODE.fd"],
        "armhfp": ["/usr/share/AAVMF/AAVMF32_CODE.fd"],
    }.get(config.architecture, [])

    for firmware in FIRMWARE_LOCATIONS:
        if os.path.exists(firmware):
            logging.warning("Couldn't find OVMF firmware blob with secure boot support, "
                            "falling back to OVMF firmware blobs without secure boot support.")
            return Path(firmware), False

    # If we can't find an architecture specific path, fall back to some generic paths that might also work.

    FIRMWARE_LOCATIONS = [
        "/usr/share/edk2/ovmf/OVMF_CODE.secboot.fd",
        "/usr/share/edk2-ovmf/OVMF_CODE.secboot.fd",  # GENTOO:
        "/usr/share/qemu/OVMF_CODE.secboot.fd",
        "/usr/share/ovmf/OVMF.secboot.fd",
        "/usr/share/OVMF/OVMF_CODE.secboot.fd",
    ]

    for firmware in FIRMWARE_LOCATIONS:
        if os.path.exists(firmware):
            return Path(firmware), True

    FIRMWARE_LOCATIONS = [
        "/usr/share/edk2/ovmf/OVMF_CODE.fd",
        "/usr/share/edk2-ovmf/OVMF_CODE.fd",  # GENTOO:
        "/usr/share/qemu/OVMF_CODE.fd",
        "/usr/share/ovmf/OVMF.fd",
        "/usr/share/OVMF/OVMF_CODE.fd",
    ]

    for firmware in FIRMWARE_LOCATIONS:
        if os.path.exists(firmware):
            logging.warn("Couldn't find OVMF firmware blob with secure boot support, "
                         "falling back to OVMF firmware blobs without secure boot support.")
            return Path(firmware), False

    die("Couldn't find OVMF UEFI firmware blob.")


def find_ovmf_vars(config: MkosiConfig) -> Path:
    OVMF_VARS_LOCATIONS = []

    if config.architecture == "x86_64":
        OVMF_VARS_LOCATIONS += ["/usr/share/ovmf/x64/OVMF_VARS.fd"]
    elif config.architecture == "i386":
        OVMF_VARS_LOCATIONS += [
            "/usr/share/edk2/ovmf-ia32/OVMF_VARS.fd",
            "/usr/share/OVMF/OVMF32_VARS_4M.fd",
        ]
    elif config.architecture == "armhfp":
        OVMF_VARS_LOCATIONS += ["/usr/share/AAVMF/AAVMF32_VARS.fd"]
    elif config.architecture == "aarch64":
        OVMF_VARS_LOCATIONS += ["/usr/share/AAVMF/AAVMF_VARS.fd"]

    OVMF_VARS_LOCATIONS += ["/usr/share/edk2/ovmf/OVMF_VARS.fd",
                            "/usr/share/edk2-ovmf/OVMF_VARS.fd",  # GENTOO:
                            "/usr/share/qemu/OVMF_VARS.fd",
                            "/usr/share/ovmf/OVMF_VARS.fd",
                            "/usr/share/OVMF/OVMF_VARS.fd"]

    for location in OVMF_VARS_LOCATIONS:
        if os.path.exists(location):
            return Path(location)

    die("Couldn't find OVMF UEFI variables file.")


@contextlib.contextmanager
def start_swtpm() -> Iterator[Optional[Path]]:

    if not shutil.which("swtpm"):
        yield None
        return

    with tempfile.TemporaryDirectory() as swtpm_state:
        swtpm_sock = Path(swtpm_state) / Path("sock")

        cmd = ["swtpm",
               "socket",
               "--tpm2",
               "--tpmstate", f"dir={swtpm_state}",
               "--ctrl", f"type=unixio,path={swtpm_sock}",
         ]

        swtpm_proc = spawn(cmd)

        try:
            yield swtpm_sock
        finally:
            swtpm_proc.wait()


@contextlib.contextmanager
def vsock_notify_handler() -> Iterator[tuple[str, dict[str, str]]]:
    """
    This yields a vsock address and a dict that will be filled in with the notifications from the VM. The
    dict should only be accessed after the context manager has been finalized.
    """
    with socket.socket(socket.AF_VSOCK, socket.SOCK_SEQPACKET) as vsock:
        vsock.bind((socket.VMADDR_CID_ANY, -1))
        vsock.listen()
        vsock.setblocking(False)

        messages = {}

        async def notify() -> None:
            loop = asyncio.get_running_loop()

            try:
                while True:
                    s, _ = await loop.sock_accept(vsock)

                    for msg in (await loop.sock_recv(s, 4096)).decode().split("\n"):
                        if not msg:
                            continue

                        k, _, v = msg.partition("=")
                        messages[k] = v

            except asyncio.CancelledError:
                pass

        with MkosiAsyncioThread(notify()):
            yield f"vsock:{socket.VMADDR_CID_HOST}:{vsock.getsockname()[1]}", messages


def run_qemu(args: MkosiArgs, config: MkosiConfig) -> None:
    accel = "tcg"
    if config.qemu_kvm == ConfigFeature.enabled or (config.qemu_kvm == ConfigFeature.auto and qemu_check_kvm_support()):
        accel = "kvm"

    firmware, fw_supports_sb = find_qemu_firmware(config)
    smm = "on" if fw_supports_sb else "off"

    if config.architecture == "aarch64":
        machine = f"type=virt,accel={accel}"
    else:
        machine = f"type=q35,accel={accel},smm={smm}"

    cmdline: list[PathString] = [
        find_qemu_binary(config),
        "-machine", machine,
        "-smp", config.qemu_smp,
        "-m", config.qemu_mem,
        "-object", "rng-random,filename=/dev/urandom,id=rng0",
        "-device", "virtio-rng-pci,rng=rng0,id=rng-device0",
        "-nic", "user,model=virtio-net-pci",
    ]

    if qemu_check_vsock_support(log=True):
        cmdline += ["-device", f"vhost-vsock-pci,guest-cid={machine_cid(config)}"]

    cmdline += ["-cpu", "max"]

    if config.qemu_gui:
        cmdline += ["-vga", "virtio"]
    else:
        # -nodefaults removes the default CDROM device which avoids an error message during boot
        # -serial mon:stdio adds back the serial device removed by -nodefaults.
        cmdline += [
            "-nographic",
            "-nodefaults",
            "-chardev", "stdio,mux=on,id=console,signal=off",
            "-serial", "chardev:console",
            "-mon", "console",
        ]

    for k, v in config.credentials.items():
        cmdline += ["-smbios", f"type=11,value=io.systemd.credential.binary:{k}={base64.b64encode(v.encode()).decode()}"]
    cmdline += ["-smbios", f"type=11,value=io.systemd.stub.kernel-cmdline-extra={' '.join(config.kernel_command_line_extra)}"]

    cmdline += ["-drive", f"if=pflash,format=raw,readonly=on,file={firmware}"]

    notifications: dict[str, str] = {}

    with contextlib.ExitStack() as stack:
        if fw_supports_sb:
            ovmf_vars = stack.enter_context(tempfile.NamedTemporaryFile(prefix=".mkosi-", dir=tmp_dir()))
            copy_path(find_ovmf_vars(config), Path(ovmf_vars.name))
            cmdline += [
                "-global", "ICH9-LPC.disable_s3=1",
                "-global", "driver=cfi.pflash01,property=secure,value=on",
                "-drive", f"file={ovmf_vars.name},if=pflash,format=raw",
            ]

        if config.ephemeral:
            f = stack.enter_context(tempfile.NamedTemporaryFile(prefix=".mkosi-", dir=config.output_dir))
            fname = Path(f.name)

            # So on one hand we want CoW off, since this stuff will
            # have a lot of random write accesses. On the other we
            # want the copy to be snappy, hence we do want CoW. Let's
            # ask for both, and let the kernel figure things out:
            # let's turn off CoW on the file, but start with a CoW
            # copy. On btrfs that works: the initial copy is made as
            # CoW but later changes do not result in CoW anymore.

            run(["chattr", "+C", fname], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            copy_path(config.output_dir / config.output, fname)
        else:
            fname = config.output_dir / config.output

        # Debian images fail to boot with virtio-scsi, see: https://github.com/systemd/mkosi/issues/725
        if config.output_format == OutputFormat.cpio:
            kernel = config.output_dir / config.output_split_kernel
            if not kernel.exists() and "-kernel" not in args.cmdline:
                die("No kernel found, please install a kernel in the cpio or provide a -kernel argument to mkosi qemu")
            cmdline += ["-kernel", kernel,
                        "-initrd", fname,
                        "-append", " ".join(config.kernel_command_line + config.kernel_command_line_extra)]
        if config.distribution == Distribution.debian:
            cmdline += ["-drive", f"if=virtio,id=hd,file={fname},format=raw"]
        else:
            cmdline += ["-drive", f"if=none,id=hd,file={fname},format=raw",
                        "-device", "virtio-scsi-pci,id=scsi",
                        "-device", "scsi-hd,drive=hd,bootindex=1"]

        swtpm_socket = stack.enter_context(start_swtpm())
        if swtpm_socket is not None:
            cmdline += ["-chardev", f"socket,id=chrtpm,path={swtpm_socket}",
                        "-tpmdev", "emulator,id=tpm0,chardev=chrtpm"]

            if config.architecture == "x86_64":
                cmdline += ["-device", "tpm-tis,tpmdev=tpm0"]
            elif config.architecture == "aarch64":
                cmdline += ["-device", "tpm-tis-device,tpmdev=tpm0"]

        addr, notifications = stack.enter_context(vsock_notify_handler())
        cmdline += ["-smbios", f"type=11,value=io.systemd.credential:vmm.notify_socket={addr}"]

        cmdline += config.qemu_args
        cmdline += args.cmdline

        run(cmdline, stdin=sys.stdin, stdout=sys.stdout, env=os.environ, log=False)

    if "EXIT_STATUS" in notifications:
        raise subprocess.CalledProcessError(int(notifications["EXIT_STATUS"]), cmdline)
