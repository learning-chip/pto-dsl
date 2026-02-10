#include "pto/pto-inst.hpp"
using namespace pto;

template <typename To, typename From>
static inline To ptoas_bitcast(From from) {
  static_assert(sizeof(To) == sizeof(From), "ptoas_bitcast: size mismatch");
  To to;
  __builtin_memcpy(&to, &from, sizeof(To));
  return to;
}

__global__ AICORE void vec_add_kernel_2d_dynamic(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, int32_t v4, int32_t v5) {
  unsigned v6 = 1;
  unsigned v7 = 0;
  int32_t v8 = 32;
  int32_t v9 = 1;
  int64_t v10 = 0;
  int64_t v11 = 4096;
  int64_t v12 = 8192;
  using T = float;
  unsigned v13 = (unsigned) v8;
  unsigned v14 = v7 * v13;
  unsigned v15 = v7 + v14;
  unsigned v16 = (unsigned) v9;
  unsigned v17 = v7 * v16;
  unsigned v18 = v15 + v17;
  __gm__ float* v19 = v1 + v18;
  using GTShape_81637264 = pto::Shape<1, 1, 1, 32, 32>;
  using GTStride_81637264 = pto::Stride<1024, 1024, 1024, 32, 1>;
  constexpr pto::Layout GT_81637264_layout = pto::Layout::ND;
  GTShape_81637264 v20 = GTShape_81637264();
  GTStride_81637264 v21 = GTStride_81637264();
  using GT_81637264 = GlobalTensor<float, GTShape_81637264, GTStride_81637264, GT_81637264_layout>;
  GT_81637264 v22 = GT_81637264(v19, v20, v21);
  unsigned v23 = (unsigned) v8;
  unsigned v24 = v7 * v23;
  unsigned v25 = v7 + v24;
  unsigned v26 = (unsigned) v9;
  unsigned v27 = v7 * v26;
  unsigned v28 = v25 + v27;
  __gm__ float* v29 = v2 + v28;
  using GTShape_81883392 = pto::Shape<1, 1, 1, 32, 32>;
  using GTStride_81883392 = pto::Stride<1024, 1024, 1024, 32, 1>;
  constexpr pto::Layout GT_81883392_layout = pto::Layout::ND;
  GTShape_81883392 v30 = GTShape_81883392();
  GTStride_81883392 v31 = GTStride_81883392();
  using GT_81883392 = GlobalTensor<float, GTShape_81883392, GTStride_81883392, GT_81883392_layout>;
  GT_81883392 v32 = GT_81883392(v29, v30, v31);

  #if defined(__DAV_VEC__)
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null> v33 = Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null>(v4, v5);
  TASSIGN(v33, v10);
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, 32, 32, SLayout::NoneBox, 512, PadValue::Null> v34;
  TASSIGN(v34, v11);
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, 32, 32, SLayout::NoneBox, 512, PadValue::Null> v35;
  TASSIGN(v35, v12);
  TLOAD(v33, v22);
  TLOAD(v34, v32);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TADD(v35, v33, v34);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  unsigned v36 = (unsigned) v8;
  unsigned v37 = v7 * v36;
  unsigned v38 = v7 + v37;
  unsigned v39 = (unsigned) v9;
  unsigned v40 = v7 * v39;
  unsigned v41 = v38 + v40;
  __gm__ float* v42 = v3 + v41;
  using GTShape_81883568 = pto::Shape<1, 1, 1, 32, 32>;
  using GTStride_81883568 = pto::Stride<1024, 1024, 1024, 32, 1>;
  constexpr pto::Layout GT_81883568_layout = pto::Layout::ND;
  GTShape_81883568 v43 = GTShape_81883568();
  GTStride_81883568 v44 = GTStride_81883568();
  using GT_81883568 = GlobalTensor<float, GTShape_81883568, GTStride_81883568, GT_81883568_layout>;
  GT_81883568 v45 = GT_81883568(v42, v43, v44);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v45, v35);
  pipe_barrier(PIPE_ALL);
  #endif // __DAV_VEC__

  return;
}


