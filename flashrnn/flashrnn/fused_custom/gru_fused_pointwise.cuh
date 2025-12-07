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
static_assert(FLASHRNN_NUM_GATES_W == 3, "Input-based gates must be 4");
static_assert(FLASHRNN_NUM_GATES_R == 3, "Recurrent gates must be 4");

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
  const auto rraw = raw_gates[1 * gates_stride];
  const auto zraw = raw_gates[2 * gates_stride];
  const auto nraw = raw_gates[3 * gates_stride];
  const auto one = dscalar_one<FLASHRNN_DTYPE_A>();
  const auto zero = dscalar_zero<FLASHRNN_DTYPE_A>();

  const auto y_cur = float2type<FLASHRNN_DTYPE_A>(type2float(states_local[0]));

  const auto rgate = sigmoid_g(rraw);
  const auto zgate = sigmoid_g(zraw);

  if (Training) {
    gates_r_inout[0 * gates_r_inout_stride] =
        float2type<FLASHRNN_DTYPE_G>(type2float(graw));
    gates_r_inout[1 * gates_r_inout_stride] =
        float2type<FLASHRNN_DTYPE_G>(type2float(rgate));
    gates_r_inout[2 * gates_r_inout_stride] =
        float2type<FLASHRNN_DTYPE_G>(type2float(zgate));

    // not needed as gates are the same for W and R
    gates_i_inout[0 * gates_i_inout_stride] =
        float2type<FLASHRNN_DTYPE_G>(type2float(graw));
    gates_i_inout[1 * gates_i_inout_stride] =
        float2type<FLASHRNN_DTYPE_G>(type2float(rgate));
    gates_i_inout[2 * gates_i_inout_stride] =
        float2type<FLASHRNN_DTYPE_G>(type2float(zgate));
    gates_i_inout[3 * gates_i_inout_stride] =
        float2type<FLASHRNN_DTYPE_G>(type2float(nraw));
  }
  const auto n_new = tanh_g(add_g(nraw, mul_g(rgate, graw)));

  auto y_new = add_g(mul_g(zgate, y_cur), mul_g(sub_g(one, zgate), n_new));

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
  const auto graw = g_i[0 * g_i_stride];
  const auto rgate = g_i[1 * g_i_stride];
  const auto zgate = g_i[2 * g_i_stride];
  const auto nraw = g_i[3 * g_i_stride];

  const auto y_cur = s[0 * s_stride];
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

  const auto dy_i = mul_g(zgate, dy_total);

  const auto nval = tanh_g(add_g(nraw, mul_g(rgate, graw)));
  const auto dg_z =
      mul_g(d_sigmoid_g(zgate), mul_g(dy_total, sub_g(y_cur, nval)));
  const auto dg_n = mul_g(dy_total, mul_g(sub_g(one, zgate), d_tanh_g(nval)));
  const auto dg_r = mul_g(mul_g(dg_n, graw), d_sigmoid_g(rgate));
  const auto dg_g = mul_g(dg_n, rgate);

  ds_inout[0] = dy_i;

  dg_r_out[0 * dg_r_out_stride] = dg_g;
  dg_r_out[1 * dg_r_out_stride] = dg_r;
  dg_r_out[2 * dg_r_out_stride] = dg_z;

  dg_i_out[0 * dg_r_out_stride] = dg_g;
  dg_i_out[1 * dg_r_out_stride] = dg_r;
  dg_i_out[2 * dg_r_out_stride] = dg_z;
  dg_i_out[2 * dg_r_out_stride] = dg_n;
}
