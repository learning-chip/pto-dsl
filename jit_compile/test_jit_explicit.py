import torch
import torch_npu

from test_ir_builder import build_module
from ptodsl.jit import compile_module, load_lib


def test_e2e_jit(launch_kernel=True):
    module = build_module()
    lib_path = compile_module(module)
    func = load_lib(lib_path)

    if launch_kernel:
        device = "npu:1"
        torch.npu.set_device(device)

        dtype = torch.float32
        shape = [32, 32]  # shape hard-coded as the kernel
        torch.manual_seed(0)
        x = torch.rand(shape, device=device, dtype=dtype) - 0.5
        y = torch.zeros(shape, device=device, dtype=dtype)

        func(x, y)
        torch.npu.synchronize()

        y_ref = torch.nn.functional.relu(x)
        torch.testing.assert_close(y, y_ref)
        print("result equal!")


if __name__ == "__main__":
    test_e2e_jit()
