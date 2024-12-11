#include "../util/cuda_error.h"
#include "../util/inline_ops.cuh"
#include <cublas_v2.h>

#include "flashrnn.h"
#include "flashrnn_pointwise.cuh"

#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>

#ifndef FLASHRNN_NUM_GATES_T
#define FLASHRNN_NUM_GATES_R 1
#define FLASHRNN_NUM_GATES_W 1
#define FLASHRNN_NUM_GATES_I 1
#define FLASHRNN_NUM_GATES_T 1
#define FLASHRNN_NUM_STATES 1
#define FLASHRNN_GRADIENT_RECURRENT_CLIPVAL 0.
#define FLASHRNN_GRADIENT_RECURRENT_CLIPVAL_VALID false
#endif

static_assert(FLASHRNN_NUM_GATES_T == 1, "Total gates must be 1");
static_assert(FLASHRNN_NUM_GATES_I == 1, "Interacting gates must be 1");
static_assert(FLASHRNN_NUM_GATES_W == 1, "Input-based gates must be 1");
static_assert(FLASHRNN_NUM_GATES_R == 1, "Recurrent gates must be 1");

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
                               // used if autoraining==true)
    FLASHRNN_DTYPE_G *g_i_out) {

  // We're in column-major order here, so increase x => increase row.
  const int row = blockDim.x * blockIdx.x + threadIdx.x; // hidden
  const int col = blockDim.y * blockIdx.y + threadIdx.y; // batch
  const int head_dim = hidden_dim / num_heads;
  const int head_idx = (blockDim.z * blockIdx.z + threadIdx.z) * head_dim;

  if (row >= head_dim || col >= batch_dim)
    return;

  // Base index into the Wx and Ry matrices.
  const int weight_idx = col * (hidden_dim * FLASHRNN_NUM_GATES_R) + row +
                         FLASHRNN_NUM_GATES_R * head_idx;

  // Base index into the output matrix. autohis is different from `weight_idx`
  // because the number of rows are different between the two sets of matrices.
  const int output_idx = col * hidden_dim + row + head_idx;

  const int g_idx = weight_idx + 0. * head_dim;

  const auto y_new = tanh_g(add_g(
      type2float(Wx[g_idx]),
      add_g(type2float(Ry[g_idx]),
            type2float(
                b[row + FLASHRNN_NUM_GATES_T * head_idx + 0 * head_dim]))));
  // Compile-time constant branch should be eliminated by compiler so we
  // have straight-through code.
  if (Training) {
    g_r_out[g_idx] = float2type<FLASHRNN_DTYPE_G>(y_new);
  }

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

  const int stride4_base_idx = col * (hidden_dim * FLASHRNN_NUM_GATES_R) + row +
                               FLASHRNN_NUM_GATES_R * head_idx;
  const int g_idx = stride4_base_idx + 0 * head_dim;

  const auto y_new = type2float(g_r[g_idx]);

  const auto dg = mul_g(d_tanh_g(y_new), dy_total);

  ds_inout[base_idx + 0 * ds_inout_stride] = float2type<FLASHRNN_DTYPE_S>(0.);

  dg_r_out[g_idx] = float2type<FLASHRNN_DTYPE_G>(dg);
}

FLASHRNN_POST_DEFINITIONS

} // namespace flashrnn
