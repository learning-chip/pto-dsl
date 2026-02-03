"""JIT compile PTO modules to shared libraries and load them for execution."""

import os
import subprocess
from pathlib import Path
import ctypes

import torch

from ptodsl.edit_cpp import convert


def compile_module(module, clean_up=True, timeout=20):
    ir_path = "./temp.pto"  # TODO: use Python `tempfile` module
    raw_cpp_path = "./temp_generated.cpp"
    edited_cpp_path = "./temp_edited.cpp"
    lib_path = "./temp_lib.so"

    # step 1, Python -> IR
    with open(ir_path, "w") as f:
        f.write(str(module))  # TODO: a direct `module.dump(path)` API?

    # step 2, IR -> CPP
    # TODO: use `ptoas --enable-insert-sync` so no need for explicit sync in frontend
    # need https://github.com/zhangstevenunity/PTOAS/issues/10
    subprocess.run(
        ["ptoas", ir_path, "-o", raw_cpp_path],
        timeout=timeout, stderr=subprocess.DEVNULL
    )

    # Step 3, preprocess cpp source
    # TODO: should extend `ptoas` emitc to largely replace this ad-doc editing
    content = Path(raw_cpp_path).read_text(encoding="utf-8")
    converted = convert(content)
    Path(edited_cpp_path).write_text(converted, encoding="utf-8")

    # Step 4, cpp -> so
    PTO_LIB_PATH = os.environ["PTO_LIB_PATH"]

    flags = [
        "-fPIC",
        "-shared",
        "-xcce",
        "--npu-arch=dav-2201",
        "-DMEMORY_BASE",  # here hardcoded for A2A3; TODO: expose this option to jit interface
        "-O2",
        "-std=c++17",
        f"-I{PTO_LIB_PATH}/include",
    ]

    subprocess.run(
        ["bisheng", *flags, edited_cpp_path, "-o", lib_path],
        timeout=timeout
    )

    if clean_up:
        os.remove(ir_path)
        os.remove(raw_cpp_path)
        os.remove(edited_cpp_path)

    return lib_path


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def load_lib(lib_path, clean_up=True):
    import torch_npu

    lib = ctypes.CDLL(lib_path)

    default_block_dim = 1  # TODO: extend kernel to multi-core

    def func_wrapper(
        x,
        y,
        block_dim=default_block_dim,
        stream=None
    ):
        if stream is None:
            stream = torch.npu.current_stream()
        # TODO (important): matching call signature to arg list information in Python `build_module`
        lib.call_kernel(
            block_dim,
            stream._as_parameter_,
            torch_to_ctypes(x),
            torch_to_ctypes(y)
        )

    if clean_up:
        os.remove(lib_path)

    return func_wrapper
