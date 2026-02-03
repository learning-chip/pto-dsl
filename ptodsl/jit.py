"""JIT compile PTO modules to shared libraries and load them for execution."""

import os
import subprocess
import sys
from pathlib import Path
import ctypes
import functools

import torch

from ptodsl import ir_builder, register_function
from ptodsl.edit_cpp import convert


def _call_meta_and_capture_env(meta_fn):
    """Run meta_fn() and capture its local namespace (for types etc.). Returns (return_value, env dict)."""
    env = {}

    def trace(frame, event, arg):
        if event == "return":
            env.clear()
            env.update(frame.f_locals)
        return trace

    old_trace = sys.gettrace()
    sys.settrace(trace)
    try:
        result = meta_fn()
    finally:
        sys.settrace(old_trace)
    return result, env


def pto_meta_data(f):
    """Decorator that marks a function as the meta-data provider (types, config) for jit_compile."""
    return f


def jit_compile(meta_data=None):
    """Decorator: build module from the kernel using meta_data env, compile to a shared lib, and replace with the loaded callable.
    meta_data() is run inside the same ir_builder() scope as the kernel so type construction has an MLIR context.
    Use string annotations for types from meta (e.g. x: \"ptr_type\") so they are resolved from meta_data's env."""
    def decorator(kernel_fn):
        compiled_func = None

        @functools.wraps(kernel_fn)
        def wrapper(*args, **kwargs):
            nonlocal compiled_func
            if compiled_func is None:
                with ir_builder() as module:
                    _result, env = _call_meta_and_capture_env(meta_data)
                    kernel_globals = {**kernel_fn.__globals__, **env}
                    kernel_with_env = type(kernel_fn)(
                        kernel_fn.__code__,
                        kernel_globals,
                        kernel_fn.__name__,
                        kernel_fn.__defaults__,
                        kernel_fn.__closure__,
                    )
                    ann = kernel_fn.__annotations__
                    resolved = {
                        k: env[v] if isinstance(v, str) and v in env else v
                        for k, v in ann.items()
                    }
                    kernel_with_env.__annotations__ = resolved
                    register_function(kernel_with_env)
                lib_path = compile_module(module)
                compiled_func = load_lib(lib_path)
            return compiled_func(*args, **kwargs)

        return wrapper

    return decorator


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
