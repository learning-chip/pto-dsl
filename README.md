# PTO-DSL

Frontend for https://github.com/zhangstevenunity/PTOAS

## ptodsl package (high-level DSL)

The `ptodsl` package provides a simplified API so that scripts like `jit_compile/mock_interface.py` behave the same as the low-level MLIR builder (e.g. `experiment/relu/relu_builder.py`). Install from repo root (requires MLIR Python bindings and PTO dialect in the environment):

```bash
pip install -e ./ptodsl
```

See `ptodsl/README.md` for usage.

## Environment

Use the docker image in https://github.com/zhangstevenunity/PTOAS/tree/main/docker

Tesed on commit on 2026/02/03 https://github.com/zhangstevenunity/PTOAS/commit/2c8d83ea0ea1ca748ce53d178bef6878e644d113


## Major TODO

- [ ] Matching lib call signature to the arg list of original Python function before jit
- [ ] Add control flow examples (mapping Python `if`, `else`, `for` to `scf` blocks), by clean-up https://github.com/zhangstevenunity/PTOAS/pull/13
- [ ] Multi-core kernel examples, ref https://github.com/zhangstevenunity/PTOAS/issues/11
- [ ] Auto-sync pass, https://github.com/zhangstevenunity/PTOAS/issues/10
