"""
High-level PTO DSL language: types, constants, and ops that lower to MLIR.
Uses the current builder context from ir_builder().
"""

import contextvars
from mlir.ir import F32Type, IndexType
from mlir.dialects import arith, pto

_builder = contextvars.ContextVar("_builder", default=None)


def _get():
    b = _builder.get()
    if b is None:
        raise RuntimeError("Not inside ir_builder() context")
    return b


# --- Type helpers (string -> attr) ---
def _blayout(s):
    return pto.BLayoutAttr.get(getattr(pto.BLayout, s))


def _slayout(s):
    return pto.SLayoutAttr.get(getattr(pto.SLayout, s))


def _pad(s):
    return pto.PadValueAttr.get(getattr(pto.PadValue, s))


def _addr_space(s):
    return pto.AddressSpaceAttr.get(getattr(pto.AddressSpace, s))


# --- Scalar / dtype: pto.float32 resolved via __getattr__ (module attrs don't use __get__) ---
def __getattr__(name):
    if name == "float32":
        return F32Type.get()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# --- Types ---
def PtrType(dtype):
    return pto.PtrType.get(dtype)


def TensorType(rank=2, dtype=None):
    return pto.TensorViewType.get(shape_or_rank=rank, element_type=dtype)


def SubTensorType(shape, dtype):
    return pto.PartitionTensorViewType.get(shape=shape, element_type=dtype)


def TileBufConfig(blayout="RowMajor", slayout="NoneBox", s_fractal_size=512, pad="Null"):
    bl = _blayout(blayout)
    sl = _slayout(slayout)
    pd = _pad(pad)
    return pto.TileBufConfigAttr.get(
        blayout=bl, slayout=sl, s_fractal_size=s_fractal_size, pad=pd
    )


def TileBufType(shape, dtype, memory_space="UB", config=None, valid_shape=None):
    if valid_shape is None:
        valid_shape = shape
    ub = _addr_space(memory_space)
    return pto.TileBufType.get(
        shape=shape,
        element_type=dtype,
        memory_space=ub,
        valid_shape=valid_shape,
        config=config,
    )


# --- Constants (pipe/event attrs; descriptors so "pto.PIPE_MTE2" works) ---
class _PipeEventAttr:
    def __init__(self, name):
        self.name = name

    def __get__(self, obj, owner=None):
        return _get().attrs[self.name]


PIPE_MTE2 = _PipeEventAttr("PIPE_MTE2")
PIPE_V = _PipeEventAttr("PIPE_V")
PIPE_MTE3 = _PipeEventAttr("PIPE_MTE3")
EVENT_ID0 = _PipeEventAttr("EVENT_ID0")


# --- Index constant ---
def const(value):
    return arith.ConstantOp(IndexType.get(), value).result


# --- Ops ---
def as_tensor(tensor_type, ptr, shape, strides):
    return pto.MakeTensorViewOp(tensor_type, ptr=ptr, shape=shape, strides=strides).result


def slice_view(subtensor_type, source, offsets, sizes):
    return pto.PartitionViewOp(
        subtensor_type, source=source, offsets=offsets, sizes=sizes
    ).result


def alloc_tile(tile_type):
    return pto.AllocTileOp(tile_type).result


def load(source_view, tile_buf):
    pto.TLoadOp(None, source_view, tile_buf)


def set_flag(pipe_from, pipe_to, event):
    pto.SetFlagOp(pipe_from, pipe_to, event)


def wait_flag(pipe_from, pipe_to, event):
    pto.WaitFlagOp(pipe_from, pipe_to, event)


def relu(input_tile, output_tile):
    pto.TReluOp(input_tile, output_tile)


def store(tile_buf, dest_view):
    pto.TStoreOp(None, tile_buf, dest_view)


# Expose _builder for __init__.py (set/reset from ir_builder)
class _BuilderVar:
    def set(self, ctx):
        return _builder.set(ctx)

    def get(self):
        return _builder.get()

    def reset(self, token):
        _builder.reset(token)


_builder_var = _BuilderVar()
