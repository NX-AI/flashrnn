#pragma once
#ifndef _FLASHRNN_POINTWISE_INCLUDED
#define _FLASHRNN_POINTWISE_INCLUDED
#endif

#include "flashrnn.h"

// needed for Functions that are of general type g = Wx + f(Ry) + b

__device__ __forceinline__ float FLASHRNNRecurrentActivation(float Ry,
                                                             uint index);

template <typename KPar, bool>
__device__ __forceinline__ void FLASHRNNPointwiseForward(
    FLASHRNN_DTYPE_S *states_local, FLASHRNN_DTYPE_A *raw_gates,
    const uint gates_stride, FLASHRNN_DTYPE_S *new_state_y,
    FLASHRNN_DTYPE_S *new_state_other, const uint new_states_stride,
    FLASHRNN_DTYPE_G *gates_r_inout, const uint gates_r_inout_stride,
    FLASHRNN_DTYPE_G *gates_i_inout, const uint gates_i_inout_stride);

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
    const uint dg_b_out_stride);
