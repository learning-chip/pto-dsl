from ptodsl import ir_builder, register_function
import ptodsl.language as pto


def build_module():
    with ir_builder() as module:
        # common, reusable type declarations
        # TODO: infer required type info from function body, or pre-declare all common meta types
        dtype = pto.float32
        ptr_type = pto.PtrType(dtype)
        tensor_type = pto.TensorType(rank=2, dtype=dtype)
        subtensor_type = pto.SubTensorType(shape=[32, 32], dtype=dtype)  # TODO: omit shape https://github.com/zhangstevenunity/PTOAS/issues/31
        tile_cfg = pto.TileBufConfig(
            blayout="RowMajor", slayout="NoneBox", s_fractal_size=512, pad="Null")
        tile_type = pto.TileBufType(
            shape=[32, 32], dtype=dtype, memory_space="UB", config=tile_cfg)
        # NOTE: valid_shape=shape if not specified

        const = pto.const
        PIPE_MTE2 = pto.PIPE_MTE2
        PIPE_MTE3 = pto.PIPE_MTE3
        PIPE_V = pto.PIPE_V
        EVENT_ID = pto.EVENT_ID0

        @register_function
        def sync_kernel_2d(x_ptr: ptr_type, y_ptr: ptr_type):
            c0 = const(0)
            c1 = const(1)
            c32 = const(32)

            tv0 = pto.as_tensor(tensor_type, ptr=x_ptr, shape=[c32, c32], strides=[c32, c1])
            tv1 = pto.as_tensor(tensor_type, ptr=y_ptr, shape=[c32, c32], strides=[c32, c1])

            # TODO: overload __getitem__
            sv0 = pto.slice_view(subtensor_type, source=tv0, offsets=[c0, c0], sizes=[c32, c32])
            sv1 = pto.slice_view(subtensor_type, source=tv1, offsets=[c0, c0], sizes=[c32, c32])

            tb0 = pto.alloc_tile(tile_type)
            tb1 = pto.alloc_tile(tile_type)

            pto.load(sv0, tb0)
            pto.set_flag(PIPE_MTE2, PIPE_V, EVENT_ID)
            pto.wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID)

            pto.relu(tb0, tb1)
            pto.set_flag(PIPE_V, PIPE_MTE3, EVENT_ID)
            pto.wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID)

            pto.store(tb1, sv1)
            # default to `return None`

    return module


if __name__ == "__main__":
    print(build_module())
