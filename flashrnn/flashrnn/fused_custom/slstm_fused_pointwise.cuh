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
  const auto m_cur = float2type<FLASHRNN_DTYPE_A>(type2float(states_local[3]));
  const auto logfplusm = add_g(logsigmoid_g(fraw), m_cur);
  const auto zval = tanh_g(raw_gates[2 * gates_stride]);
  const auto oraw = raw_gates[3 * gates_stride];
  const auto one = dscalar_one<FLASHRNN_DTYPE_A>();
  const auto zero = dscalar_zero<FLASHRNN_DTYPE_A>();

  const auto c_cur = float2type<FLASHRNN_DTYPE_A>(type2float(states_local[1]));
  auto n_cur = float2type<FLASHRNN_DTYPE_A>(type2float(states_local[2]));

  FLASHRNN_DTYPE_A m_new;
  if (eq_zero_g(n_cur)) {
    m_new = iraw;
#ifdef FLASHRNN_STABILIZATION_EPSILON_FIRST
    n_cur = add_g(n_cur, float2type<FLASHRNN_DTYPE_S>(
                             FLASHRNN_STABILIZATION_EPSILON_FIRST));
#endif
  } else {
    m_new = max_g(iraw, logfplusm);
  }
#ifdef FLASHRNN_STABILIZATION_EPSILON
  n_cur = add_g(n_cur,
                float2type<FLASHRNN_DTYPE_S>(FLASHRNN_STABILIZATION_EPSILON));
#endif
  const auto igate = min_g(one, exp_g(sub_g(iraw, m_new)));
  const auto fgate = min_g(one, exp_g(sub_g(logfplusm, m_new)));
  const auto ogate = sigmoid_g(oraw);

  if (Training) {
    gates_r_inout[0 * gates_r_inout_stride] =
        float2type<FLASHRNN_DTYPE_G>(type2float(igate));
    gates_r_inout[1 * gates_r_inout_stride] =
        float2type<FLASHRNN_DTYPE_G>(type2float(fraw));
    gates_r_inout[2 * gates_r_inout_stride] =
        float2type<FLASHRNN_DTYPE_G>(type2float(zval));
    gates_r_inout[3 * gates_r_inout_stride] =
        float2type<FLASHRNN_DTYPE_G>(type2float(ogate));

    // not needed as gates are the same for W and R
    // gates_i_inout[0 * gates_i_inout_stride] = igate;
    // gates_i_inout[1 * gates_i_inout_stride] = fraw;
    // gates_i_inout[2 * gates_i_inout_stride] = zval;
    // gates_i_inout[3 * gates_i_inout_stride] = ogate;
  }
  const auto c_new = add_g(mul_g(fgate, c_cur), mul_g(igate, zval));
  // n cannot get smaller than one by construction, just numerically there might
  // be problems sometimes
  auto n_new = max_g(add_g(mul_g(fgate, n_cur), igate), one);

#ifdef FLASHRNN_STABILIZATION_EPSILON_NEW
  n_new = add_g(
      n_new, float2type<FLASHRNN_DTYPE_S>(FLASHRNN_STABILIZATION_EPSILON_NEW));
#endif

  auto y_new = mul_g(ogate, div_g(c_new, n_new));

#if FLASHRNN_FORWARD_CLIPVAL_VALID
  y_new = clip_val_g(
      y_new,
      neg_g(float2type<FLASHRNN_DTYPE_S>((float)FLASHRNN_FORWARD_CLIPVAL)),
      float2type<FLASHRNN_DTYPE_S>((float)FLASHRNN_FORWARD_CLIPVAL));
#endif

  states_local[0] = float2type<FLASHRNN_DTYPE_S>(type2float(y_new));
  states_local[1] = float2type<FLASHRNN_DTYPE_S>(type2float(c_new));
  states_local[2] = float2type<FLASHRNN_DTYPE_S>(type2float(n_new));
  states_local[3] = float2type<FLASHRNN_DTYPE_S>(type2float(m_new));

  new_state_y[0] = states_local[0];
  new_states_other[0 * new_states_stride] = states_local[1];
  new_states_other[1 * new_states_stride] = states_local[2];
  new_states_other[2 * new_states_stride] = states_local[3];
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
  const auto fraw = g_r[1 * g_r_stride];
  const auto zval = g_r[2 * g_r_stride];
  const auto ogate = g_r[3 * g_r_stride];

  // const auto y_cur = s[0 * s_stride];
  const auto c_cur = s[1 * s_stride];
  const auto n_cur = s[2 * s_stride];
  const auto m_cur = s[3 * s_stride];
  const auto m_new = s_new[3 * s_new_stride];
  const auto n_new = s_new[2 * s_new_stride];
  const auto y_new = s_new[0 * s_new_stride];

  const auto zero = dscalar_zero<FLASHRNN_DTYPE_S>();
  const auto one = dscalar_one<FLASHRNN_DTYPE_S>();

  const auto logfplusm = add_g(logsigmoid_g(fraw), m_cur);
  const auto fgate = min_g(one, exp_g(sub_g(logfplusm, m_new)));
  const auto fsig = min_g(one, sigmoid_g(fraw));

#if (FLASHRNN_GRADIENT_RECURRENT_CLIPVAL_VALID)
  ds_inout[0] = clip_val_g(
      ds_inout[0],
      neg_g(float2type<FLASHRNN_DTYPE_S>(
          (float)FLASHRNN_GRADIENT_RECURRENT_CLIPVAL)),
      float2type<FLASHRNN_DTYPE_S>((float)FLASHRNN_GRADIENT_RECURRENT_CLIPVAL));
  if ((isnan_g(ds_inout[0])) || (isinf_g(ds_inout[0]))) {
    ds_inout[0] = dscalar_zero<FLASHRNN_DTYPE_S>();
  }
#endif

  auto dy_total = add_g(ds_new[0 * ds_new_stride], ds_inout[0]);

  auto dy_inter = div_g(dy_total, n_new);
  const auto dc_total = add_g(
      ds_inout[1], add_g(ds_new[1 * ds_new_stride], mul_g(dy_inter, ogate)));
  const auto dn_total = add_g(
      ds_inout[2], sub_g(ds_new[2 * ds_new_stride], mul_g(dy_inter, y_new)));

  const auto dg_i = mul_g(igate, add_g(mul_g(zval, dc_total), dn_total));
  const auto dg_f =
      mul_g(mul_g(fgate, add_g(mul_g(dc_total, c_cur), mul_g(dn_total, n_cur))),
            sub_g(one, fsig));
  const auto dg_z = mul_g(mul_g(dc_total, igate), d_tanh_g(zval));
  const auto dg_o = mul_g(sub_g(one, ogate), mul_g(y_new, dy_total));

  const auto dc_i = mul_g(fgate, dc_total);
  const auto dn_i = mul_g(fgate, dn_total);

  ds_inout[0] = zero;
  ds_inout[1] = dc_i;
  ds_inout[2] = dn_i;
  ds_inout[3] = zero;

  dg_r_out[0 * dg_r_out_stride] = dg_i;
  dg_r_out[1 * dg_r_out_stride] = dg_f;
  dg_r_out[2 * dg_r_out_stride] = dg_z;
  dg_r_out[3 * dg_r_out_stride] = dg_o;
}
