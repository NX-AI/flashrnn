#pragma once

#include "flashrnn.h"

namespace flashrnn {

template <bool>
__global__ void FLASHRNNPointwiseForward(
    const int batch_dim, const int hidden_dim, const int num_heads,
    const FLASHRNN_DTYPE_G *Wx, // Precomputed (Wx) vector
    const FLASHRNN_DTYPE_G *Ry, // Precomputed (Ry) vector
    const FLASHRNN_DTYPE_B *b,  // Bias for gates
    const FLASHRNN_DTYPE_S *s,  // Input  state
    const uint s_stride,
    FLASHRNN_DTYPE_S *s_out, // Output recurrent state
    const uint s_out_stride,
    FLASHRNN_DTYPE_G *g_r_out, // Output vector v (Wx + Ry + b) (only
                               // used if autoraining==true)
    FLASHRNN_DTYPE_G *g_i_out);

__global__ void FLASHRNNPointwiseBackward(
    const int batch_dim, const int hidden_dim, const int num_heads,
    const FLASHRNN_DTYPE_S *s, const uint s_stride, const FLASHRNN_DTYPE_G *g_r,
    const FLASHRNN_DTYPE_G *g_i, const FLASHRNN_DTYPE_B *bias,
    const FLASHRNN_DTYPE_S *s_new, const uint s_new_stride,
    const FLASHRNN_DTYPE_S *ds_new, const uint ds_new_stride,
    FLASHRNN_DTYPE_S *ds_inout, const uint ds_inout_stride,
    FLASHRNN_DTYPE_G *dg_r_out, FLASHRNN_DTYPE_G *dg_i_out,
    FLASHRNN_DTYPE_G *dg_b_out);

} // namespace flashrnn

#define FLASHRNN_POST_DEFINITIONS                                              \
  template __global__ void FLASHRNNPointwiseForward<true>(                     \
      const int batch_dim, const int hidden_dim, const int num_heads,          \
      const FLASHRNN_DTYPE_G *Wx, const FLASHRNN_DTYPE_G *Ry,                  \
      const FLASHRNN_DTYPE_B *b, const FLASHRNN_DTYPE_S *s,                    \
      const uint s_stride, FLASHRNN_DTYPE_S *s_out, const uint s_out_stride,   \
      FLASHRNN_DTYPE_G *g_r_out, FLASHRNN_DTYPE_G *g_i_out);                   \
  template __global__ void FLASHRNNPointwiseForward<false>(                    \
      const int batch_dim, const int hidden_dim, const int num_heads,          \
      const FLASHRNN_DTYPE_G *Wx, const FLASHRNN_DTYPE_G *Ry,                  \
      const FLASHRNN_DTYPE_B *b, const FLASHRNN_DTYPE_S *s,                    \
      const uint s_stride, FLASHRNN_DTYPE_S *s_out, const uint s_out_stride,   \
      FLASHRNN_DTYPE_G *g_r_out, FLASHRNN_DTYPE_G *g_i_out);                   \
  __global__ void FLASHRNNPointwiseBackward(                                   \
      const int batch_dim, const int hidden_dim, const int num_heads,          \
      const FLASHRNN_DTYPE_S *s, const uint s_stride,                          \
      const FLASHRNN_DTYPE_G *g_r, const FLASHRNN_DTYPE_G *g_i,                \
      const FLASHRNN_DTYPE_B *b, const FLASHRNN_DTYPE_S *s_new,                \
      const uint s_new_stride, const FLASHRNN_DTYPE_S *ds_new,                 \
      const uint ds_new_stride, FLASHRNN_DTYPE_S *ds_inout,                    \
      const uint ds_inout_stride, FLASHRNN_DTYPE_G *dg_r_out,                  \
      FLASHRNN_DTYPE_G *dg_i_out, FLASHRNN_DTYPE_G *dg_b_out);
