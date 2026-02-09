import ctypes
import torch
import torch_npu


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def lib_to_func(lib):
    def add_func(
        out,
        src0,
        src1,
        stream_ptr=None
        ):

        if stream_ptr is None:
            # make sure stream is lazy-queued after `torch.npu.set_device`
            stream_ptr = torch.npu.current_stream()._as_parameter_

        dtype = out.dtype
        if out.dtype == torch.float32:
            lib.call_kernel_fp32(
                stream_ptr,
                torch_to_ctypes(out),
                torch_to_ctypes(src0),
                torch_to_ctypes(src1)
            )
        elif out.dtype == torch.float16:
            lib.call_kernel_fp16(
                stream_ptr,
                torch_to_ctypes(out),
                torch_to_ctypes(src0),
                torch_to_ctypes(src1)
            )
        elif out.dtype == torch.int32:
            lib.call_kernel_int32(
                stream_ptr,
                torch_to_ctypes(out),
                torch_to_ctypes(src0),
                torch_to_ctypes(src1)
            )
        else:
            raise ValueError
    return add_func


def test_add(add_func, dtype=torch.float32):
    shape = [64, 64]  # shape hard-coded as the kernel
    torch.manual_seed(0)
    device = "npu"
    src0 = torch.rand(shape, device=device, dtype=dtype)
    src1 = torch.rand(shape, device=device, dtype=dtype)
    out = torch.empty(shape, device=device, dtype=dtype)

    add_func(out, src0, src1)
    torch.npu.synchronize()

    out_ref = src0 + src1
    torch.testing.assert_close(out, out_ref)
    print("result equal!")

if __name__ == "__main__":
    lib_path = "./add.so"
    lib = ctypes.CDLL(lib_path)
    add_func = lib_to_func(lib)

    device = "npu:1"
    torch.npu.set_device(device)

    test_add(add_func, dtype=torch.float32)
    test_add(add_func, dtype=torch.float16)
    test_add(add_func, dtype=torch.int32)
