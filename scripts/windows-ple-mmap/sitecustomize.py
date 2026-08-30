"""Native-Windows bridge for the Desktop-assisted PR #279 source checkout.

The loopback TCP, Selector-loop, installed-kernel, and Windows compiler approach
builds on prior Windows work in FreeToken PR #232. This module is loaded by
Python automatically when its directory is first on PYTHONPATH.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


if os.name == "nt":
    import asyncio

    # PyZMQ needs add_reader(), which the Windows Proactor loop does not provide.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Uvicorn 0.35+ can bypass the policy and create Proactor directly.
    import uvicorn.loops.asyncio as _uvicorn_asyncio

    _uvicorn_asyncio.asyncio_loop_factory = (
        lambda use_subprocess=False: asyncio.SelectorEventLoop
    )

    # Reuse matching Windows DLLs and CUDA sources supplied by FreeToken Desktop.
    import freetoken.kernel

    installed_kernel = Path(sys.prefix) / "Lib" / "site-packages" / "freetoken" / "kernel"
    if installed_kernel.is_dir() and str(installed_kernel) not in freetoken.kernel.__path__:
        freetoken.kernel.__path__.append(str(installed_kernel))

    import freetoken.kernel.utils as _kernel_utils

    installed_csrc = installed_kernel / "csrc"
    if installed_csrc.is_dir():
        _kernel_utils.KERNEL_PATH = installed_csrc
        _kernel_utils.DEFAULT_INCLUDE = [str(installed_csrc / "include")]

    cuda_root = os.environ.get("CUDA_PATH")
    if not cuda_root:
        from torch.utils.cpp_extension import CUDA_HOME

        cuda_root = CUDA_HOME
    if cuda_root:
        cudart = Path(cuda_root) / "lib" / "x64" / "cudart.lib"
        if cudart.is_file():
            _kernel_utils.DEFAULT_LDFLAGS = [f'"{cudart}"']

    # POSIX page-cache advice is optional and does not exist on Windows.
    if not hasattr(os, "POSIX_FADV_DONTNEED"):
        os.POSIX_FADV_DONTNEED = 4
    if not hasattr(os, "posix_fadvise"):
        os.posix_fadvise = lambda fd, offset, length, advice: None

    from freetoken.scheduler.config import SchedulerConfig
    from freetoken.server.args import ServerArgs

    def _scheduler_addr(offset: int):
        return property(lambda self: f"tcp://127.0.0.1:{self.server_port + offset}")

    # Port +1 is FreeToken's distributed link; internal ZMQ links use +2..+6.
    SchedulerConfig.zmq_backend_addr = _scheduler_addr(2)
    SchedulerConfig.zmq_detokenizer_addr = _scheduler_addr(3)
    SchedulerConfig.zmq_scheduler_broadcast_addr = _scheduler_addr(4)
    ServerArgs.zmq_frontend_addr = _scheduler_addr(5)
    ServerArgs.zmq_tokenizer_addr = property(
        lambda self: self.zmq_detokenizer_addr
        if self.share_tokenizer
        else f"tcp://127.0.0.1:{self.server_port + 6}"
    )

    # apache-tvm-ffi 0.1.13 emits C++17 and split MSVC host flags on Windows.
    # FreeToken headers require C++20, while NVCC needs the two host flags packed.
    import tvm_ffi.cpp.extension as _tvm_extension

    if not getattr(_tvm_extension, "_freetoken_windows_flags_fixed", False):
        _original_generate_ninja = _tvm_extension._generate_ninja_build

        def _generate_ninja_windows(*args, **kwargs):
            ninja = _original_generate_ninja(*args, **kwargs)
            ninja = ninja.replace("/std:c++17", "/std:c++20")
            return ninja.replace(
                "-Xcompiler /std:c++20 /O2",
                "-Xcompiler=/std:c++20,/O2",
            )

        _tvm_extension._generate_ninja_build = _generate_ninja_windows
        _tvm_extension._freetoken_windows_flags_fixed = True
