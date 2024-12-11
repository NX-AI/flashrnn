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
#define FLASHRNN_NUM_GATES_R 4
#define FLASHRNN_NUM_GATES_W 4
#define FLASHRNN_NUM_GATES_I 4
#define FLASHRNN_NUM_GATES_T 4
#define FLASHRNN_GRADIENT_RECURRENT_CLIPVAL 0.
#define FLASHRNN_GRADIENT_RECURRENT_CLIPVAL_VALID false
#endif

static_assert(FLASHRNN_NUM_GATES_T == 4, "Total gates must be 4");
static_assert(FLASHRNN_NUM_GATES_I == 4, "Interacting gates must be 4");
static_assert(FLASHRNN_NUM_GATES_W == 4, "Input-based gates must be 4");
static_assert(FLASHRNN_NUM_GATES_R == 4, "Recurrent gates must be 4");

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
  const auto iraw = raw_gates[0 * gates_stride];
  const auto fraw = raw_gates[1 * gates_stride];
  const auto zraw = raw_gates[2 * gates_stride];
  const auto oraw = raw_gates[3 * gates_stride];
  const auto one = dscalar_one<FLASHRNN_DTYPE_A>();
  const auto zero = dscalar_zero<FLASHRNN_DTYPE_A>();

  const auto c_cur = float2type<FLASHRNN_DTYPE_A>(type2float(states_local[1]));

  const auto igate = sigmoid_g(iraw);
  const auto fgate = sigmoid_g(fraw);
  const auto zval = tanh_g(zraw);
  const auto ogate = sigmoid_g(oraw);

  if (Training) {
    gates_r_inout[0 * gates_r_inout_stride] =
        float2type<FLASHRNN_DTYPE_G>(type2float(igate));
    gates_r_inout[1 * gates_r_inout_stride] =
        float2type<FLASHRNN_DTYPE_G>(type2float(fgate));
    gates_r_inout[2 * gates_r_inout_stride] =
        float2type<FLASHRNN_DTYPE_G>(type2float(zval));
    gates_r_inout[3 * gates_r_inout_stride] =
        float2type<FLASHRNN_DTYPE_G>(type2float(ogate));

    // not needed as gates are the same for W and R
    // gates_i_inout[0 * gates_i_inout_stride] = igate;
    // gates_i_inout[1 * gates_i_inout_stride] = fraw;
    // gates_i_inout[2 * gates_i_inout_stride] = zraw;
    // gates_i_inout[3 * gates_i_inout_stride] = ogate;
  }
  const auto c_new = add_g(mul_g(fgate, c_cur), mul_g(igate, zval));

  auto y_new = mul_g(ogate, tanh_g(c_new));

#if FLASHRNN_FORWARD_CLIPVAL_VALID
  y_new = clip_val_g(
      y_new,
      neg_g(float2type<FLASHRNN_DTYPE_S>((float)FLASHRNN_FORWARD_CLIPVAL)),
      float2type<FLASHRNN_DTYPE_S>((float)FLASHRNN_FORWARD_CLIPVAL));
#endif

  states_local[0] = float2type<FLASHRNN_DTYPE_S>(type2float(y_new));
  states_local[1] = float2type<FLASHRNN_DTYPE_S>(type2float(c_new));

  new_state_y[0] = states_local[0];
  new_states_other[0 * new_states_stride] = states_local[1];
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
  const auto igate = g_r[0 * g_r_stride];
  const auto fgate = g_r[1 * g_r_stride];
  const auto zval = g_r[2 * g_r_stride];
  const auto ogate = g_r[3 * g_r_stride];

  // const auto y_cur = s[0 * s_stride];
  const auto c_cur = s[1 * s_stride];
  // not needed
  // const auto y_new = s_new[0 * s_new_stride];
  const auto c_new = s_new[1 * s_new_stride];
  const auto c_new_tanh = tanh_g(c_new);

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
  auto dc_total = add_g(ds_new[1 * ds_new_stride], ds_inout[1]);

  const auto dc_tanh = mul_g(ogate, dy_total);
  dc_total = add_g(dc_total, mul_g(d_tanh_g(c_new_tanh), dc_tanh));

  const auto dg_i = mul_g(d_sigmoid_g(igate), mul_g(zval, dc_total));
  const auto dg_f = mul_g(d_sigmoid_g(fgate), mul_g(c_cur, dc_total));
  const auto dg_z = mul_g(mul_g(dc_total, igate), d_tanh_g(zval));
  const auto dg_o = mul_g(d_sigmoid_g(ogate), mul_g(c_new_tanh, dy_total));

  const auto dc_i = mul_g(fgate, dc_total);

  ds_inout[0] = zero;
  ds_inout[1] = dc_i;

  dg_r_out[0 * dg_r_out_stride] = dg_i;
  dg_r_out[1 * dg_r_out_stride] = dg_f;
  dg_r_out[2 * dg_r_out_stride] = dg_z;
  dg_r_out[3 * dg_r_out_stride] = dg_o;
}
