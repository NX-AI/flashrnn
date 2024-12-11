#include "../util/cuda_error.h"
#include "../util/inline_ops.cuh"
#include <cublas_v2.h>

#include "flashrnn.h"
#include "flashrnn_pointwise.cuh"

#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>

#ifndef FLASHRNN_NUM_GATES_T
#define FLASHRNN_NUM_GATES_R 3
#define FLASHRNN_NUM_GATES_W 3
#define FLASHRNN_NUM_GATES_I 4
#define FLASHRNN_NUM_GATES_T 4
#define FLASHRNN_NUM_STATES 2
#define FLASHRNN_GRADIENT_RECURRENT_CLIPVAL 0.
#define FLASHRNN_GRADIENT_RECURRENT_CLIPVAL_VALID false
#endif

static_assert(FLASHRNN_NUM_GATES_T == 4, "Total gates must be 4");
static_assert(FLASHRNN_NUM_GATES_I == 4, "Interacting gates must be 4");
static_assert(FLASHRNN_NUM_GATES_W == 3, "Input-based gates must be 3");
static_assert(FLASHRNN_NUM_GATES_R == 3, "Recurrent gates must be 3");

namespace flashrnn {

template <bool Training>
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
                               // used if training==true)
    FLASHRNN_DTYPE_G *g_i_out) {

  // We're in column-major order here, so increase x => increase row.
  const int row = blockDim.x * blockIdx.x + threadIdx.x; // hidden
  const int col = blockDim.y * blockIdx.y + threadIdx.y; // batch
  const int head_dim = hidden_dim / num_heads;
  const int head_idx = (blockDim.z * blockIdx.z + threadIdx.z) * head_dim;

  if (row >= head_dim || col >= batch_dim)
    return;

  // Base index into the Wx and Ry matrices. Both FLASHRNN_NUM_GATES_R ==
  // FLASHRNN_NUM_GATES_W = 3
  const int stride3_base_idx = col * (hidden_dim * FLASHRNN_NUM_GATES_R) + row +
                               FLASHRNN_NUM_GATES_R * head_idx;
  const int stride4_base_idx = col * (hidden_dim * FLASHRNN_NUM_GATES_I) + row +
                               FLASHRNN_NUM_GATES_I * head_idx;

  // Base index into the output matrix. autohis is different from `weight_idx`
  // because the number of rows are different between the two sets of matrices.
  const int output_idx = col * hidden_dim + row + head_idx;

  const int g_idx_r = stride3_base_idx + 0. * head_dim;
  const int r_idx_r = stride3_base_idx + 1. * head_dim;
  const int z_idx_r = stride3_base_idx + 2. * head_dim;

  const int r_idx_w = stride3_base_idx + 0. * head_dim;
  const int z_idx_w = stride3_base_idx + 1. * head_dim;
  const int n_idx_w = stride3_base_idx + 2. * head_dim;

  const auto y_cur = type2float(s[output_idx + 0 * s_stride]);
  const auto graw = add_g(
      type2float(Ry[g_idx_r]),
      type2float(b[row + FLASHRNN_NUM_GATES_T * head_idx + 0 * head_dim]));
  const auto rraw = add_g(
      type2float(Wx[r_idx_w]),
      add_g(
          type2float(Ry[r_idx_r]),
          type2float(b[row + FLASHRNN_NUM_GATES_T * head_idx + 1 * head_dim])));
  const auto zraw = add_g(
      type2float(Wx[z_idx_w]),
      add_g(
          type2float(Ry[z_idx_r]),
          type2float(b[row + FLASHRNN_NUM_GATES_T * head_idx + 2 * head_dim])));
  const auto nraw = add_g(
      type2float(Wx[n_idx_w]),
      type2float(b[row + FLASHRNN_NUM_GATES_T * head_idx + 3 * head_dim]));
  float one = 1.;

  const auto rgate = sigmoid_g(rraw);
  const auto zgate = sigmoid_g(zraw);

  // Compile-time constant branch should be eliminated by compiler so we have
  // straight-through code.
  if (Training) {
    g_r_out[g_idx_r] = float2type<FLASHRNN_DTYPE_G>(graw);
    g_r_out[r_idx_r] = float2type<FLASHRNN_DTYPE_G>(rgate);
    g_r_out[z_idx_r] = float2type<FLASHRNN_DTYPE_G>(zgate);

    g_i_out[stride4_base_idx + 0 * head_dim] =
        float2type<FLASHRNN_DTYPE_G>(graw);
    g_i_out[stride4_base_idx + 1 * head_dim] =
        float2type<FLASHRNN_DTYPE_G>(rgate);
    g_i_out[stride4_base_idx + 2 * head_dim] =
        float2type<FLASHRNN_DTYPE_G>(zgate);
    g_i_out[stride4_base_idx + 3 * head_dim] =
        float2type<FLASHRNN_DTYPE_G>(nraw);
  }

  const auto n_new = tanh_g(add_g(nraw, mul_g(rgate, graw)));
  auto y_new = add_g(mul_g(zgate, y_cur), mul_g(sub_g(one, zgate), n_new));

#if FLASHRNN_FORWARD_CLIPVAL_VALID
  y_new = clip_val_g(y_new, neg_g((float)FLASHRNN_FORWARD_CLIPVAL),
                     (float)FLASHRNN_FORWARD_CLIPVAL);
#endif

  s_out[output_idx + 0 * s_out_stride] = float2type<FLASHRNN_DTYPE_S>(y_new);
}

__global__ void FLASHRNNPointwiseBackward(
    const int batch_dim, const int hidden_dim, const int num_heads,
    const FLASHRNN_DTYPE_S *s, const uint s_stride, const FLASHRNN_DTYPE_G *g_r,
    const FLASHRNN_DTYPE_G *g_i,
    const FLASHRNN_DTYPE_B *b, // Bias for gates
    const FLASHRNN_DTYPE_S *s_new, const uint s_new_stride,
    const FLASHRNN_DTYPE_S *ds_new, const uint ds_new_stride,
    FLASHRNN_DTYPE_S *ds_inout, const uint ds_inout_stride,
    FLASHRNN_DTYPE_G *dg_r_out, FLASHRNN_DTYPE_G *dg_i_out,
    FLASHRNN_DTYPE_G *dg_b_out) {
  const int row = blockDim.x * blockIdx.x + threadIdx.x; // hidden
  const int col = blockDim.y * blockIdx.y + threadIdx.y; // batch
  const int head_dim = hidden_dim / num_heads;
  const int head_idx = (blockDim.z * blockIdx.z + threadIdx.z) * head_dim;

  if (row >= head_dim || col >= batch_dim)
    return;

  const int base_idx = col * hidden_dim + row + head_idx;
  auto dy_recurrent = type2float(ds_inout[base_idx + 0 * ds_inout_stride]);

#if (FLASHRNN_GRADIENT_RECURRENT_CLIPVAL_VALID)
  dy_recurrent = clip_val_g(dy_recurrent,
                            neg_g((float)FLASHRNN_GRADIENT_RECURRENT_CLIPVAL),
                            (float)FLASHRNN_GRADIENT_RECURRENT_CLIPVAL);
#endif
  const auto dy_total =
      add_g(type2float(ds_new[base_idx + 0 * ds_new_stride]), dy_recurrent);
  auto dc_total = add_g(type2float(ds_new[base_idx + 1 * ds_new_stride]),
                        type2float(ds_inout[base_idx + 1 * ds_inout_stride]));

  const int stride3_base_idx = col * (hidden_dim * FLASHRNN_NUM_GATES_R) + row +
                               FLASHRNN_NUM_GATES_R * head_idx;
  const int stride4_base_idx = col * (hidden_dim * FLASHRNN_NUM_GATES_I) + row +
                               FLASHRNN_NUM_GATES_I * head_idx;

  const int g_idx_r = stride3_base_idx + 0. * head_dim;
  const int r_idx_r = stride3_base_idx + 1. * head_dim;
  const int z_idx_r = stride3_base_idx + 2. * head_dim;

  const int g_idx_i = stride4_base_idx + 0 * head_dim;
  const int r_idx_i = stride4_base_idx + 1 * head_dim;
  const int z_idx_i = stride4_base_idx + 2 * head_dim;
  const int n_idx_i = stride4_base_idx + 3 * head_dim;

  const auto graw = type2float(g_i[g_idx_i]);
  const auto rgate = type2float(g_i[r_idx_i]);
  const auto zgate = type2float(g_i[z_idx_i]);
  const auto nraw = type2float(g_i[n_idx_i]);

  const float one = 1.;
  const auto y_cur = type2float(s[base_idx + 0 * s_stride]);

  const auto dy_prev = mul_g(zgate, dy_total);

  const auto nval = tanh_g(add_g(nraw, mul_g(rgate, graw)));
  const auto dg_z =
      mul_g(d_sigmoid_g(zgate), mul_g(dy_total, sub_g(y_cur, nval)));
  const auto dg_n = mul_g(dy_total, mul_g(sub_g(one, zgate), d_tanh_g(nval)));
  const auto dg_r = mul_g(mul_g(dg_n, graw), d_sigmoid_g(rgate));
  const auto dg_g = mul_g(dg_n, rgate);

  ds_inout[base_idx + 0 * ds_inout_stride] =
      float2type<FLASHRNN_DTYPE_S>(dy_prev);

  dg_r_out[g_idx_r] = float2type<FLASHRNN_DTYPE_G>(dg_g);
  dg_r_out[r_idx_r] = float2type<FLASHRNN_DTYPE_G>(dg_r);
  dg_r_out[z_idx_r] = float2type<FLASHRNN_DTYPE_G>(dg_z);

  dg_i_out[g_idx_i] = float2type<FLASHRNN_DTYPE_G>(dg_g);
  dg_i_out[r_idx_i] = float2type<FLASHRNN_DTYPE_G>(dg_r);
  dg_i_out[z_idx_i] = float2type<FLASHRNN_DTYPE_G>(dg_z);
  dg_i_out[n_idx_i] = float2type<FLASHRNN_DTYPE_G>(dg_n);
}

FLASHRNN_POST_DEFINITIONS

} // namespace flashrnn
