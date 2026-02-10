from contextlib import contextmanager

from mlir.dialects import arith, pto
from mlir.ir import F32Type, IndexType, InsertionPoint, IntegerType


def _unwrap(value):
    if isinstance(value, Value):
        return value.raw
    if isinstance(value, list):
        return [_unwrap(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_unwrap(v) for v in value)
    return value


class Value:
    def __init__(self, raw):
        self.raw = raw

    def __mul__(self, other):
        return Value(arith.MulIOp(_unwrap(self), _unwrap(other)).result)

    def __rmul__(self, other):
        return Value(arith.MulIOp(_unwrap(other), _unwrap(self)).result)

    def __add__(self, other):
        return Value(arith.AddIOp(_unwrap(self), _unwrap(other)).result)

    def __radd__(self, other):
        return Value(arith.AddIOp(_unwrap(other), _unwrap(self)).result)

    def __getattr__(self, item):
        return getattr(self.raw, item)


def wrap_value(value):
    if isinstance(value, Value):
        return value
    return Value(value)


float32 = F32Type.get()
int32 = IntegerType.get_signless(32)


def PtrType(dtype):
    return pto.PtrType.get(dtype)


def TensorType(*, rank, dtype):
    return pto.TensorViewType.get(rank, dtype)


def SubTensorType(*, shape, dtype):
    return pto.PartitionTensorViewType.get(shape, dtype)


class TileBufConfig:
    def __init__(self, blayout="RowMajor", slayout="NoneBox", s_fractal_size=512, pad="Null"):
        self._bl = pto.BLayoutAttr.get(getattr(pto.BLayout, blayout))
        self._sl = pto.SLayoutAttr.get(getattr(pto.SLayout, slayout))
        self._pd = pto.PadValueAttr.get(getattr(pto.PadValue, pad))
        self._s_fractal_size = s_fractal_size

    @property
    def attr(self):
        return pto.TileBufConfigAttr.get(self._bl, self._sl, self._s_fractal_size, self._pd)


def TileBufType(*, shape, valid_shape, dtype, memory_space, config):
    space = pto.AddressSpaceAttr.get(getattr(pto.AddressSpace, memory_space))
    cfg = config.attr if isinstance(config, TileBufConfig) else config
    return pto.TileBufType.get(shape, dtype, space, valid_shape, cfg)


def const(value):
    return Value(arith.ConstantOp(IndexType.get(), value).result)


def get_block_idx():
    return Value(pto.GetBlockIdxOp().result)


def get_subblock_idx():
    return Value(pto.GetSubBlockIdxOp().result)


def get_subblock_num():
    return Value(pto.GetSubBlockNumOp().result)


def index_cast(value, index_type=IndexType):
    if hasattr(index_type, "get"):
        dst = index_type.get()
    else:
        dst = index_type
    return Value(arith.IndexCastOp(dst, _unwrap(value)).result)


def as_tensor(tensor_type, *, ptr, shape, strides):
    return Value(pto.MakeTensorViewOp(tensor_type, _unwrap(ptr), _unwrap(shape), _unwrap(strides)).result)


def slice_view(subtensor_type, *, source, offsets, sizes):
    return Value(
        pto.PartitionViewOp(subtensor_type, _unwrap(source), offsets=_unwrap(offsets), sizes=_unwrap(sizes)).result
    )


@contextmanager
def vector_section():
    section = pto.SectionVectorOp()
    block = section.body.blocks.append()
    with InsertionPoint(block):
        yield


def alloc_tile(tile_type, *, valid_row, valid_col):
    return Value(pto.AllocTileOp(tile_type, valid_row=_unwrap(valid_row), valid_col=_unwrap(valid_col)).result)


def load(source, dest):
    pto.TLoadOp(None, _unwrap(source), _unwrap(dest))


def add(lhs, rhs, out):
    pto.TAddOp(_unwrap(lhs), _unwrap(rhs), _unwrap(out))


def store(source, dest):
    pto.TStoreOp(None, _unwrap(source), _unwrap(dest))
