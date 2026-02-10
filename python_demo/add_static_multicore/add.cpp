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
  int32_t v8 = 1280;
  int32_t v9 = 32;
  int32_t v10 = 1;
  int64_t v11 = 0;
  int64_t v12 = 4096;
  int64_t v13 = 8192;
  using T = float;
  int64_t v14 = get_block_idx();
  int64_t v15 = get_subblockid();
  int64_t v16 = get_subblockdim();
  uint64_t v17 = (uint64_t) v14;
  uint64_t v18 = (uint64_t) v16;
  uint64_t v19 = v17 * v18;
  int64_t v20 = (int64_t) v19;
  uint64_t v21 = (uint64_t) v20;
  uint64_t v22 = (uint64_t) v15;
  uint64_t v23 = v21 + v22;
  int64_t v24 = (int64_t) v23;
  int32_t v25 = (int32_t) v24;
  uint32_t v26 = (uint32_t) v25;
  uint32_t v27 = (uint32_t) v9;
  uint32_t v28 = v26 * v27;
  int32_t v29 = (int32_t) v28;
  unsigned v30 = (unsigned) v29;
  unsigned v31 = (unsigned) v9;
  unsigned v32 = v30 * v31;
  unsigned v33 = v7 + v32;
  unsigned v34 = (unsigned) v10;
  unsigned v35 = v7 * v34;
  unsigned v36 = v33 + v35;
  __gm__ float* v37 = v1 + v36;
  using GTShape_371383984 = pto::Shape<1, 1, 1, 32, 32>;
  using GTStride_371383984 = pto::Stride<1024, 1024, 1024, 32, 1>;
  constexpr pto::Layout GT_371383984_layout = pto::Layout::ND;
  GTShape_371383984 v38 = GTShape_371383984();
  GTStride_371383984 v39 = GTStride_371383984();
  using GT_371383984 = GlobalTensor<float, GTShape_371383984, GTStride_371383984, GT_371383984_layout>;
  GT_371383984 v40 = GT_371383984(v37, v38, v39);
  unsigned v41 = (unsigned) v29;
  unsigned v42 = (unsigned) v9;
  unsigned v43 = v41 * v42;
  unsigned v44 = v7 + v43;
  unsigned v45 = (unsigned) v10;
  unsigned v46 = v7 * v45;
  unsigned v47 = v44 + v46;
  __gm__ float* v48 = v2 + v47;
  using GTShape_371698912 = pto::Shape<1, 1, 1, 32, 32>;
  using GTStride_371698912 = pto::Stride<1024, 1024, 1024, 32, 1>;
  constexpr pto::Layout GT_371698912_layout = pto::Layout::ND;
  GTShape_371698912 v49 = GTShape_371698912();
  GTStride_371698912 v50 = GTStride_371698912();
  using GT_371698912 = GlobalTensor<float, GTShape_371698912, GTStride_371698912, GT_371698912_layout>;
  GT_371698912 v51 = GT_371698912(v48, v49, v50);
  unsigned v52 = (unsigned) v29;
  unsigned v53 = (unsigned) v9;
  unsigned v54 = v52 * v53;
  unsigned v55 = v7 + v54;
  unsigned v56 = (unsigned) v10;
  unsigned v57 = v7 * v56;
  unsigned v58 = v55 + v57;
  __gm__ float* v59 = v3 + v58;
  using GTShape_371699120 = pto::Shape<1, 1, 1, 32, 32>;
  using GTStride_371699120 = pto::Stride<1024, 1024, 1024, 32, 1>;
  constexpr pto::Layout GT_371699120_layout = pto::Layout::ND;
  GTShape_371699120 v60 = GTShape_371699120();
  GTStride_371699120 v61 = GTStride_371699120();
  using GT_371699120 = GlobalTensor<float, GTShape_371699120, GTStride_371699120, GT_371699120_layout>;
  GT_371699120 v62 = GT_371699120(v59, v60, v61);

  #if defined(__DAV_VEC__)
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null> v63 = Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null>(v4, v5);
  TASSIGN(v63, v11);
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null> v64 = Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null>(v4, v5);
  TASSIGN(v64, v12);
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null> v65 = Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null>(v4, v5);
  TASSIGN(v65, v13);
  TLOAD(v63, v40);
  TLOAD(v64, v51);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TADD(v65, v63, v64);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v62, v65);
  pipe_barrier(PIPE_ALL);
  #endif // __DAV_VEC__

  return;
}


