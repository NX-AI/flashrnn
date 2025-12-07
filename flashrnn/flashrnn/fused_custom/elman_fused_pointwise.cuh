#pragma once
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>

#ifndef _FLASHRNN_POINTWISE_INCLUDED
#define _FLASHRNN_POINTWISE_INCLUDED
#endif

#include "../util/cuda_error.h"
#include "../util/inline_ops.cuh"
#include "flashrnn.h"

#ifndef FLASHRNN_NUM_GATES_T
#define FLASHRNN_NUM_GATES_R 1
#define FLASHRNN_NUM_GATES_W 1
#define FLASHRNN_NUM_GATES_I 1
#define FLASHRNN_NUM_GATES_T 1
#define FLASHRNN_GRADIENT_RECURRENT_CLIPVAL 0.
#define FLASHRNN_GRADIENT_RECURRENT_CLIPVAL_VALID false
#endif

static_assert(FLASHRNN_NUM_GATES_T == 1, "Total gates must be 1");
static_assert(FLASHRNN_NUM_GATES_I == 1, "Interacting gates must be 1");
static_assert(FLASHRNN_NUM_GATES_W == 1, "Input-based gates must be 1");
static_assert(FLASHRNN_NUM_GATES_R == 1, "Recurrent gates must be 1");

__device__ __forceinline__ float FLASHRNNRecurrentActivation(float Ry,
                                                             uint index) {
  return Ry;
}

template <bool Training>
__device__ __forceinline__ void FLASHRNNPointwiseForward(
    FLASHRNN_DTYPE_S *states_local, const FLASHRNN_DTYPE_A *raw_gates,
    const uint gates_stride, FLASHRNN_DTYPE_S *new_state_y,
    FLASHRNN_DTYPE_S *new_states_other, const uint new_states_stride,
    FLASHRNN_DTYPE_G *gates_r_inout, const uint gates_r_inout_stride,
    FLASHRNN_DTYPE_G *gates_i_inout, const uint gates_i_inout_stride) {
  const auto graw = raw_gates[0 * gates_stride];
  const auto gval = tanh_g(graw);

  if (Training) {
    gates_r_inout[0 * gates_r_inout_stride] =
        float2type<FLASHRNN_DTYPE_G>(type2float(gval));
  }
  auto y_new = gval;

#if FLASHRNN_FORWARD_CLIPVAL_VALID
  y_new = clip_val_g(
      y_new,
      neg_g(float2type<FLASHRNN_DTYPE_S>((float)FLASHRNN_FORWARD_CLIPVAL)),
      float2type<FLASHRNN_DTYPE_S>((float)FLASHRNN_FORWARD_CLIPVAL));
#endif

  states_local[0] = float2type<FLASHRNN_DTYPE_S>(type2float(y_new));

  new_state_y[0] = states_local[0];
}

__device__ __forceinline__ void FLASHRNNPointwiseBackward(
    const FLASHRNN_DTYPE_G *g_r, const uint g_r_stride,
    const FLASHRNN_DTYPE_G *g_i, const uint g_i_stride,
    const FLASHRNN_DTYPE_S *s, const uint s_stride,
    const FLASHRNN_DTYPE_S *s_new, const uint s_new_stride,
    const FLASHRNN_DTYPE_S *ds_new, const uint ds_new_stride,
    const FLASHRNN_DTYPE_B *additional_bias_local, FLASHRNN_DTYPE_S *ds_inout,
    const uint ds_inout_stride, FLASHRNN_DTYPE_G *dg_r_out,
    const uint dg_r_out_stride, FLASHRNN_DTYPE_G *dg_i_out,
    const uint dg_i_out_stride, FLASHRNN_DTYPE_G *dg_b_out,
    const uint dg_b_out_stride) {
  const auto gval = g_r[0 * g_r_stride];

  const auto zero = dscalar_zero<FLASHRNN_DTYPE_S>();
  const auto one = dscalar_one<FLASHRNN_DTYPE_S>();

#if (FLASHRNN_GRADIENT_RECURRENT_CLIPVAL_VALID)
  ds_inout[0] = clip_val_g(
      ds_inout[0],
      float2type<FLASHRNN_DTYPE_S>(
          neg_g((float)FLASHRNN_GRADIENT_RECURRENT_CLIPVAL)),
      float2type<FLASHRNN_DTYPE_S>((float)FLASHRNN_GRADIENT_RECURRENT_CLIPVAL));
#endif

  const auto dy_total = add_g(ds_new[0 * ds_new_stride], ds_inout[0]);

  const auto dg = mul_g(dy_total, d_tanh_g(gval));

  ds_inout[0] = zero;

  dg_r_out[0 * dg_r_out_stride] = dg;
}
