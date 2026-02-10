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
  using GTShape_459189072 = pto::Shape<1, 1, 1, 32, 32>;
  using GTStride_459189072 = pto::Stride<1024, 1024, 1024, 32, 1>;
  constexpr pto::Layout GT_459189072_layout = pto::Layout::ND;
  GTShape_459189072 v20 = GTShape_459189072();
  GTStride_459189072 v21 = GTStride_459189072();
  using GT_459189072 = GlobalTensor<float, GTShape_459189072, GTStride_459189072, GT_459189072_layout>;
  GT_459189072 v22 = GT_459189072(v19, v20, v21);
  unsigned v23 = (unsigned) v8;
  unsigned v24 = v7 * v23;
  unsigned v25 = v7 + v24;
  unsigned v26 = (unsigned) v9;
  unsigned v27 = v7 * v26;
  unsigned v28 = v25 + v27;
  __gm__ float* v29 = v2 + v28;
  using GTShape_459187664 = pto::Shape<1, 1, 1, 32, 32>;
  using GTStride_459187664 = pto::Stride<1024, 1024, 1024, 32, 1>;
  constexpr pto::Layout GT_459187664_layout = pto::Layout::ND;
  GTShape_459187664 v30 = GTShape_459187664();
  GTStride_459187664 v31 = GTStride_459187664();
  using GT_459187664 = GlobalTensor<float, GTShape_459187664, GTStride_459187664, GT_459187664_layout>;
  GT_459187664 v32 = GT_459187664(v29, v30, v31);
  unsigned v33 = (unsigned) v8;
  unsigned v34 = v7 * v33;
  unsigned v35 = v7 + v34;
  unsigned v36 = (unsigned) v9;
  unsigned v37 = v7 * v36;
  unsigned v38 = v35 + v37;
  __gm__ float* v39 = v3 + v38;
  using GTShape_459187168 = pto::Shape<1, 1, 1, 32, 32>;
  using GTStride_459187168 = pto::Stride<1024, 1024, 1024, 32, 1>;
  constexpr pto::Layout GT_459187168_layout = pto::Layout::ND;
  GTShape_459187168 v40 = GTShape_459187168();
  GTStride_459187168 v41 = GTStride_459187168();
  using GT_459187168 = GlobalTensor<float, GTShape_459187168, GTStride_459187168, GT_459187168_layout>;
  GT_459187168 v42 = GT_459187168(v39, v40, v41);

  #if defined(__DAV_VEC__)
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null> v43 = Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null>(v4, v5);
  TASSIGN(v43, v10);
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null> v44 = Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null>(v4, v5);
  TASSIGN(v44, v11);
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null> v45 = Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null>(v4, v5);
  TASSIGN(v45, v12);
  TLOAD(v43, v22);
  TLOAD(v44, v32);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TADD(v45, v43, v44);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v42, v45);
  pipe_barrier(PIPE_ALL);
  #endif // __DAV_VEC__

  return;
}


