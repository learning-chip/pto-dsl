import ctypes
import torch
import torch_npu


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def load_lib(lib_path):
    # lib_path = os.path.abspath(lib_path)
    lib = ctypes.CDLL(lib_path)

    default_block_dim = 20  # 910B4, TODO: query platform information
    def add_func(x, y, z, block_dim=default_block_dim, stream_ptr=None):
        N = x.numel()
        # TODO: customize call args according to cpp `void call_kernel` signature

        if stream_ptr is None:
            stream_ptr = torch.npu.current_stream()._as_parameter_

        lib.call_kernel(
            block_dim,
            stream_ptr,
            torch_to_ctypes(x),
            torch_to_ctypes(y),
            torch_to_ctypes(z),
            N,
        )

    return add_func


def test_add():
    device = "npu:1"
    dtype = torch.float16
    torch.npu.set_device(device)

    shape = [20, 2048]
    x = torch.rand(shape, device=device, dtype=dtype)
    y = torch.rand(shape, device=device, dtype=dtype)
    z = torch.empty(shape, device=device, dtype=dtype)

    add_func =  load_lib("./add_lib.so")
    add_func(x, y, z)
    torch.npu.synchronize()

    z_ref = x + y
    torch.testing.assert_close(z, z_ref)
    print("ADD test pass!")


if __name__ == "__main__":
    test_add()
