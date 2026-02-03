# ptodsl

High-level DSL for PTO MLIR. It provides a simplified API that maps to the low-level `mlir.ir` / `mlir.dialects` (func, arith, pto) so that scripts like `jit_compile/mock_interface.py` behave the same as the original low-level builder (e.g. `experiment/relu/relu_builder.py`).

## Install

Requires the MLIR Python bindings and PTO dialect (e.g. from PTOAS or your environment). Then install ptodsl in editable mode from the repo root:

```bash
# From the pto-dsl repo root
pip install -e ./ptodsl
```

Or from inside the `ptodsl` directory:

```bash
cd ptodsl
pip install -e .
```

## Usage

```python
from ptodsl import ir_builder, register_function
import ptodsl.language as pto

def build_module():
    with ir_builder() as module:
        dtype = pto.float32
        ptr_type = pto.PtrType(dtype)
        # ... type declarations ...
        @register_function
        def sync_kernel_2d(x_ptr: ptr_type, y_ptr: ptr_type):
            # ... body using pto.const, pto.as_tensor, pto.load, etc. ...
            pass
    return module
```
