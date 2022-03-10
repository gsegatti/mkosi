# SPDX-License-Identifier: LGPL-2.1+

from __future__ import annotations

import contextlib
import copy
import os
import signal
import subprocess
import unittest
from argparse import Namespace
from textwrap import dedent
from typing import Any, Iterator, Optional, Sequence

import pexpect  # type: ignore

from . import (
    MKOSI_COMMANDS_SUDO,
    CompletedProcess,
    build_stuff,
    check_native,
    check_output,
    check_root,
    init_namespace,
    load_args,
    needs_build,
    parse_args,
    parse_boolean,
    prepend_to_environ_path,
    run_command_image,
    run_qemu_cmdline,
    run_shell_cmdline,
    unlink_output,
)
from .backend import Distribution, MkosiArgs, MkosiNotSupportedException, Verb, die


class Machine:
    def __init__(self, args: Sequence[str] = [], debug: bool = False) -> None:
        # Remains None until image is built and booted, then receives pexpect process.
        self._serial: Optional[pexpect.spawn] = None
        self.exit_code: int = -1
        self.debug = debug
        self.stack = contextlib.ExitStack()
        self.args: MkosiArgs
        self.parsed: Namespace
        self.unified: bool = False

        self.parsed = parse_args(args)["default"]

        # By default, we build every valid image by forcing the verb = Verb.build.
        # But we want to see which Verb the image was meant to use when booting later.
        # If no verb was passed via the __init__() args, we look for the environment variable.
        # By default, Mkosi makes the Verb build if nothing is passed, hence the first condition.
        if self.parsed.verb == Verb.build and Verb.build.name not in args:
            verb = os.getenv("MKOSI_TEST_DEFAULT_VERB")
            # This way, if environment variable is not set, we assign nspawn by default to test classes with no Verb.
            if verb in (Verb.boot.name, None):
                self.parsed.verb = Verb.boot
            elif verb == Verb.qemu.name:
                self.parsed.verb = Verb.qemu
            elif verb == Verb.shell.name:
                self.parsed.verb = Verb.shell
            else:
                die("No valid verb was entered.")

        # Add the arguments in the machine class itself, rather than typing this for every testing function.
        self.parsed.force = 1
        self.parsed.autologin = True
        self.parsed.ephemeral = True
        if self.parsed.verb == Verb.qemu:
            self.parsed.hostonly_initrd = True
            self.parsed.netdev = True
            self.parsed.ssh = True
        elif self.parsed.verb not in (Verb.shell, Verb.boot):
            die("No valid verb was entered.")

        # We create a copy of the results of parse_args().
        # This way, when we get to the stage of booting an image we can know if that's possible.
        # Since "self.parsed" is changed to make sure the image is built before failing to be booted.
        # We then need another variable to see if current options are a valid boot or not.
        self.parsed.bootable = False
        self.parsed.qemu_headless = False

        # We want to make sure images which are not bootable are still built.
        # Therefore, we overpass verb and with_unified_kernel_images passed by the user for now.
        # Otherwise, load_args() raises an exception before the building is done.
        # Then, when we actually boot the image, the user's verb is brought back.
        tmp = Verb.qemu
        if self.parsed.verb == Verb.boot:
            tmp = Verb.boot

        self.parsed.verb = Verb.build
        if not self.parsed.with_unified_kernel_images and "uefi" in self.parsed.boot_protocols:
            if self.parsed.distribution in (Distribution.debian.name, Distribution.ubuntu.name, Distribution.mageia.name, Distribution.opensuse.name):
                self.parsed.with_unified_kernel_images = True
                self.unified = True


        self.args = load_args(copy.deepcopy(self.parsed))


        self.parsed.verb = tmp
        if self.parsed.verb == Verb.qemu:
            self.parsed.bootable = True
            self.parsed.qemu_headless = True
        if self.unified:
            self.parsed.with_unified_kernel_images = False


        self.args.verb = self.parsed.verb

    @property
    def serial(self) -> pexpect.spawn:
        if self._serial is None:
            raise ValueError(
                        dedent(
                            """\
                            Trying to access serial console before machine boot or after machine shutdown.
                            In order to boot the machine properly, use it as a context manager.
                            Then, a Mkosi image will be booted in the __enter__ method.
                            """
                        )
                    )
        return self._serial

    def _ensure_booted(self) -> None:
        # Try to access the serial console which will raise an exception if the machine is not currently booted.
        assert self._serial is not None or self.args.verb == Verb.shell

    def build(self) -> None:
        if self.args.verb in MKOSI_COMMANDS_SUDO:
            check_root()
            unlink_output(self.args)

        if self.args.verb == Verb.build:
            check_output(self.args)

        if needs_build(self.args):
            check_root()
            check_native(self.args)
            init_namespace(self.args)
            build_stuff(self.args)

    def __enter__(self) -> Machine:
        self.build()
        self.boot()

        return self

    def boot(self) -> None:
        if self.args.verb == Verb.shell:
            return

        with contextlib.ExitStack() as stack:
            prepend_to_environ_path(self.args.extra_search_paths)

            if self.args.verb == Verb.boot:
                cmdline = run_shell_cmdline(self.args)
            elif self.args.verb == Verb.qemu:
                # We must keep the temporary file opened at run_qemu_cmdline accessible, hence the context stack.
                cmdline = stack.enter_context(run_qemu_cmdline(self.args))
            else:
                die("No valid verb was entered.")

            cmd = " ".join(str(x) for x in cmdline)

            # Here we have something equivalent to the command lines used on spawn() and run() from backend.py.
            # We use pexpect to boot an image that we will be able to interact with in the future.
            # Then we tell the process to look for the # sign, which indicates the CLI for that image is active.
            # Once we've build/boot an image the CLI will prompt "root@image ~]# ".
            # Then, when pexpects finds the '#' it means we're ready to interact with the process.
            self._serial = pexpect.spawnu(cmd, logfile=None, timeout=240)
            self._serial.expect("#")
            self.stack = stack.pop_all()

    def run(self, commands: Sequence[str], timeout: int = 900, check: bool = True) -> CompletedProcess:
        self._ensure_booted()

        process = run_command_image(self.args, commands, timeout, check, subprocess.PIPE, subprocess.PIPE)
        if self.debug:
            print(f"Stdout:\n {process.stdout}")
            print(f"Stderr:\n {process.stderr}")

        return process

    def kill(self) -> None:
        self.__exit__(None, None, None)

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        if self._serial:
            self._serial.kill(signal.SIGTERM)
            self.exit_code = self._serial.wait()
            self._serial = None
        self.stack.__exit__(*args, **kwargs)


@contextlib.contextmanager
def pytest_skip_not_supported(args: Namespace) -> Iterator[None]:
    """See if load_args() raises exception about combination of arguments parsed."""
    try:
        load_args(copy.deepcopy(args))
        yield
    except MkosiNotSupportedException as exception:
        raise unittest.SkipTest(str(exception))


class MkosiMachineTest(unittest.TestCase):
    args: Sequence[str]
    machine: Machine

    def __init_subclass__(cls, args: Sequence[str] = []) -> None:
        cls.args = args

    @classmethod
    def setUpClass(cls) -> None:
        if os.getuid() != 0:
            raise unittest.SkipTest("Must be invoked as root.")

        cls.machine = Machine(cls.args)

        verb = cls.machine.args.verb
        no_nspawn = parse_boolean(os.getenv("MKOSI_TEST_NO_NSPAWN", "0"))
        no_qemu = parse_boolean(os.getenv("MKOSI_TEST_NO_QEMU", "0"))

        if no_nspawn and verb == Verb.boot:
            raise unittest.SkipTest("Nspawn test skipped due to environment variable.")
        if no_qemu and verb == Verb.qemu:
            raise unittest.SkipTest("Qemu test skipped due to environment variable.")

        cls.machine.build()

    def setUp(self) -> None:
        # Replacing underscores which makes name invalid.
        # Necessary for shell otherwise racing conditions to the disk image will happen.
        test_name = self.id().split(".")[3]
        self.machine.args.hostname = test_name.replace("_", "-")

        print(self.machine.parsed.bootable)
        with pytest_skip_not_supported(self.machine.parsed):
            self.machine.boot()

    def tearDown(self) -> None:
        self.machine.kill()
