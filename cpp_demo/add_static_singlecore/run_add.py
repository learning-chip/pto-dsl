import ctypes
import torch
import torch_npu


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def load_lib(lib_path):
    lib = ctypes.CDLL(lib_path)

    def add_func(
        out,
        src0,
        src1,
        stream_ptr=None
        ):

        if stream_ptr is None:
            # make sure stream is lazy-queued after `torch.npu.set_device`
            stream_ptr = torch.npu.current_stream()._as_parameter_

        lib.call_kernel_int32(
            stream_ptr,
            torch_to_ctypes(out),
            torch_to_ctypes(src0),
            torch_to_ctypes(src1)
        )

    return add_func


def test_add():
    device = "npu:1"
    dtype = torch.int32
    torch.npu.set_device(device)

    shape = [64, 64]  # shape hard-coded as the kernel
    torch.manual_seed(0)
    src0 = torch.rand(shape, device=device, dtype=dtype)
    src1 = torch.rand(shape, device=device, dtype=dtype)
    out = torch.empty(shape, device=device, dtype=dtype)

    add_func = load_lib("./add.so")
    add_func(out, src0, src1)
    torch.npu.synchronize()

    out_ref = src0 + src1
    torch.testing.assert_close(out, out_ref)
    print("result equal!")

if __name__ == "__main__":
    test_add()
