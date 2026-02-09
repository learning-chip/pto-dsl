#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include "acl/acl.h"

using namespace pto;

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
__global__ AICORE void runTAdd( __gm__ T __out__ *out, __gm__ T __in__ *src0,  __gm__ T __in__ *src1) {
#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)
    using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
    using DynStridDim5 = Stride<1, 1, 1, kTCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    TileData src0Tile(vRows, vCols);
    TileData src1Tile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x10000);
    TASSIGN(dstTile, 0x20000);
    
    GlobalData src0Global(src0);
    GlobalData src1Global(src1);
    GlobalData dstGlobal(out);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TADD(dstTile, src0Tile, src1Tile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
#else // else branch for `#if defined(__DAV_C220_VEC__)`
// do nothing for Cube branch
#endif
}

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
void LaunchTAdd(T *out, T *src0, T *src1, void *stream)
{
    if constexpr ( std::is_same_v<T, aclFloat16> )
        runTAdd<half, kTRows_, kTCols_, vRows, vCols><<<1, nullptr, stream>>>(
            (half*)(out), (half*)(src0), (half*)(src1)
            );
    else 
        runTAdd<T, kTRows_, kTCols_, vRows, vCols><<<1, nullptr, stream>>>(
            out, src0, src1
            );
}

// template void LaunchTAdd<float, 64, 64, 64, 64>(float *out, float *src0, float *src1, void *stream);
// template void LaunchTAdd<int32_t, 64, 64, 64, 64>(int32_t *out, int32_t *src0, int32_t *src1, void *stream);
// template void LaunchTAdd<int16_t, 64, 64, 64, 64>(int16_t *out, int16_t *src0, int16_t *src1, void *stream);
// template void LaunchTAdd<aclFloat16, 16, 256, 16, 256>(aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1,
//     void *stream);

extern "C" void call_kernel_fp32(
    void *stream, uint8_t *out, uint8_t *src0, uint8_t *src1)
{
    // No need for blockDim here, hard-coded to 1
    LaunchTAdd<float, 64, 64, 64, 64>(
        (float *)out, (float *)src0, (float *)src1, stream);
}

extern "C" void call_kernel_int32(
    void *stream, uint8_t *out, uint8_t *src0, uint8_t *src1)
{
    // No need for blockDim here, hard-coded to 1
    LaunchTAdd<int32_t, 64, 64, 64, 64>(
        (int32_t *)out, (int32_t *)src0, (int32_t *)src1, stream);
}
