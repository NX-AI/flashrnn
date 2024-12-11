// Copyright 2024 NXAI GmbH, All Rights Reserved
// Author: Korbinian Poeppel
// Adapted from the haste library
//
// See:
// Copyright 2020 LMNT, Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>

#include "../util/blas.h"
#include "../util/cuda_error.h"
#include "../util/inline_ops.cuh"
#include "flashrnn.h"

#ifndef _FLASHRNN_POINTWISE_INCLUDED
#include "flashrnn_fused_pointwise_base.cuh"
#endif

#include <cooperative_groups.h>
#include <driver_types.h>
#include <mma.h>
#include <stdio.h>

#define CEIL_DIV(a, b) (((a) + (b)-1) / (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

// #define DEBUG
#ifdef FLASHRNN_USE_DTYPE_FLOAT32
#define DTYPE float
#define ACC_DTYPE float
#endif
#ifdef FLASHRNN_USE_DTYPE_FLOAT16
#define DTYPE __half
#define ACC_DTYPE __half
#endif
#ifdef FLASHRNN_USE_DTYPE_BFLOAT16
#define DTYPE __nv_bfloat16
#define ACC_DTYPE float
#endif

namespace {

using namespace nvcuda;

// TODO check BRTCG > 1, right now no speed increase
// #define FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_GATE 2
// #define FLASHRNN_BACKWARD_WARP_TILING_COUNT_GATE 8      // BWTCG
// #define FLASHRNN_BACKWARD_WARP_RECURRENT_CACHED_GATE 32 // BWRCG optimal
// for 1024

// offset to reduce memory bank conflicts in shared memory
#define WARP_SIZE 32 // warpSize = 32 threads

#define _NUM_BLAS_STREAMS 2
#define _FLOAT4FACTOR 8

#define _FUSED_KERNEL_MAX_THREADS                                              \
  (WARP_SIZE * FLASHRNN_BACKWARD_WARP_TILING_COUNT_GATE *                      \
   FLASHRNN_HIDDEN_SIZE / FLASHRNN_NUM_HEADS /                                 \
   FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_HIDDEN /                           \
   FLASHRNN_BACKWARD_WARP_LOOPING_COUNT_HIDDEN *                               \
   FLASHRNN_BACKWARD_WARP_TILING_COUNT_BATCH /                                 \
   FLASHRNN_BACKWARD_WARP_TILING_DIM_HIDDEN)

#define _FUSED_KERNEL_MIN_BLOCKS                                               \
  (FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_HIDDEN *                           \
   FLASHRNN_BACKWARD_BLOCK_TILING_COUNT_BATCH *                                \
   FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_GATE *                             \
   FLASHRNN_BACKWARD_MULTIHEAD_TILING_COUNT)

__global__ void __launch_bounds__(_FUSED_KERNEL_MAX_THREADS,
                                  _FUSED_KERNEL_MIN_BLOCKS)
    FLASHRNNCellFusedBackwardCut(
        const uint steps, const uint batch_dim,
        const FLASHRNN_DTYPE_R *R, // recurrect matrix head_dim x head_dim [H,
                                   // FLASHRNN_NUM_GATES_R * H]
        const FLASHRNN_DTYPE_B
            *b, // Bias for gates [G, FLASHRNN_NUM_GATES_T * H]
        const FLASHRNN_DTYPE_S *states, // states [S, T + 1, B, H]
        FLASHRNN_DTYPE_G
            *g_r_inout, // Output activations (Wx + Ry + b) [], also
                        // contains gate values [T, G-1, B, H] other gates
        FLASHRNN_DTYPE_G
            *g_i_inout,        // [FLASHRNN_NUM_GATES_T, T, B, H]?  input gate
        FLASHRNN_DTYPE_G *g_b, // [FLASHRNN_NUM_GATES_T, T, B, H]?  input gate
        const FLASHRNN_DTYPE_S *d_states_ext, FLASHRNN_DTYPE_S *d_states) {

  uint rgate_offset =
      FLASHRNN_NUM_GATES_R * FLASHRNN_HIDDEN_SIZE * batch_dim * (steps - 1);
  uint igate_offset =
      FLASHRNN_NUM_GATES_I * FLASHRNN_HIDDEN_SIZE * batch_dim * (steps - 1);
  uint bgate_offset = (FLASHRNN_NUM_GATES_T - FLASHRNN_NUM_GATES_I) *
                      FLASHRNN_HIDDEN_SIZE * batch_dim * (steps - 1);
  uint state_offset = (steps - 1) * FLASHRNN_HIDDEN_SIZE * batch_dim;

  const uint global_state_idx = (blockIdx.x * blockDim.x + threadIdx.x);
  const uint global_batch_idx = (blockIdx.y * blockDim.y + threadIdx.y);

  FLASHRNN_DTYPE_S ds_inout_local[FLASHRNN_NUM_STATES];

  if ((global_batch_idx < batch_dim) &&
      (global_state_idx < FLASHRNN_HIDDEN_SIZE)) {
#pragma unroll
    for (uint sub_state_idx = 0; sub_state_idx < FLASHRNN_NUM_STATES;
         sub_state_idx++) {
      ds_inout_local[sub_state_idx] =
          d_states[FLASHRNN_HIDDEN_SIZE * global_batch_idx + global_state_idx +
                   sub_state_idx * FLASHRNN_HIDDEN_SIZE * batch_dim];
    }

    for (uint t = steps; t >= 1; t--) {

      FLASHRNNPointwiseBackward(
          g_r_inout + rgate_offset +
              FLASHRNN_NUM_GATES_R * FLASHRNN_HIDDEN_SIZE * global_batch_idx +
              FLASHRNN_NUM_GATES_R * global_state_idx,
          1,
          g_i_inout + igate_offset +
              FLASHRNN_NUM_GATES_I * FLASHRNN_HIDDEN_SIZE * global_batch_idx +
              FLASHRNN_NUM_GATES_I * global_state_idx,
          1,
          states + state_offset + FLASHRNN_HIDDEN_SIZE * global_batch_idx +
              global_state_idx,
          FLASHRNN_HIDDEN_SIZE * batch_dim * (steps + 1),
          states + state_offset + FLASHRNN_HIDDEN_SIZE * batch_dim +
              FLASHRNN_HIDDEN_SIZE * global_batch_idx + global_state_idx,
          FLASHRNN_HIDDEN_SIZE * batch_dim * (steps + 1),
          d_states_ext + state_offset + FLASHRNN_HIDDEN_SIZE * batch_dim +
              FLASHRNN_HIDDEN_SIZE * global_batch_idx + global_state_idx,
          FLASHRNN_HIDDEN_SIZE * batch_dim * (steps + 1),
          b + global_state_idx * FLASHRNN_NUM_GATES_T + FLASHRNN_NUM_GATES_I,
          ds_inout_local, 1,
          g_r_inout + rgate_offset +
              FLASHRNN_NUM_GATES_R * FLASHRNN_HIDDEN_SIZE * global_batch_idx +
              FLASHRNN_NUM_GATES_R * global_state_idx,
          1,
          g_i_inout + igate_offset +
              FLASHRNN_NUM_GATES_I * FLASHRNN_HIDDEN_SIZE * global_batch_idx +
              FLASHRNN_NUM_GATES_I * global_state_idx,
          1,
          g_b + bgate_offset +
              (FLASHRNN_NUM_GATES_T - FLASHRNN_NUM_GATES_I) *
                  (FLASHRNN_HIDDEN_SIZE * global_batch_idx + global_state_idx),
          1);
      rgate_offset -= FLASHRNN_NUM_GATES_R * FLASHRNN_HIDDEN_SIZE * batch_dim;
      igate_offset -= FLASHRNN_NUM_GATES_I * FLASHRNN_HIDDEN_SIZE * batch_dim;
      state_offset -= FLASHRNN_HIDDEN_SIZE * batch_dim;
      bgate_offset -= (FLASHRNN_NUM_GATES_T - FLASHRNN_NUM_GATES_I) *
                      FLASHRNN_HIDDEN_SIZE * batch_dim;

    } // end of time loop
#pragma unroll
    for (uint sub_state_idx = 0; sub_state_idx < FLASHRNN_NUM_STATES;
         sub_state_idx++) {
      d_states[FLASHRNN_HIDDEN_SIZE * global_batch_idx + global_state_idx +
               sub_state_idx * FLASHRNN_HIDDEN_SIZE * batch_dim] =
          ds_inout_local[sub_state_idx];
    }
  }
}

__global__ void gradientBiasAggregationKernel(
    const uint hidden_size, const uint batch_size, const uint num_heads,
    const uint steps, const FLASHRNN_DTYPE_G *gate_gradients_i,
    const FLASHRNN_DTYPE_G *gate_gradients_bias_only, FLASHRNN_DTYPE_B *db) {
  uint idx = threadIdx.x + blockDim.x * blockIdx.x;
  uint gate_idx = idx % FLASHRNN_NUM_GATES_T;

  if (idx < FLASHRNN_NUM_GATES_T * hidden_size) {
    if (gate_idx < FLASHRNN_NUM_GATES_I) {
      float res = 0.;
      for (uint t = 0; t < steps; t++) {
        for (uint b = 0; b < batch_size; b++) {
          res = add_g(res,
                      type2float<FLASHRNN_DTYPE_G>(
                          gate_gradients_i[(t * batch_size + b) * hidden_size *
                                               FLASHRNN_NUM_GATES_I +
                                           FLASHRNN_NUM_GATES_I *
                                               (idx / FLASHRNN_NUM_GATES_T) +
                                           gate_idx]));
        }
      }
      atomicAdd(db + idx, float2type<FLASHRNN_DTYPE_B>(res));
    } else if (idx < FLASHRNN_NUM_GATES_T * hidden_size) {
      float res = 0.;
      for (uint t = 0; t < steps; t++) {
        for (uint b = 0; b < batch_size; b++) {
          res = add_g(
              res, type2float<FLASHRNN_DTYPE_G>(
                       gate_gradients_bias_only
                           [(t * batch_size + b) * hidden_size *
                                (FLASHRNN_NUM_GATES_T - FLASHRNN_NUM_GATES_I) +
                            (idx / FLASHRNN_NUM_GATES_T) *
                                (FLASHRNN_NUM_GATES_T - FLASHRNN_NUM_GATES_I) +
                            gate_idx - FLASHRNN_NUM_GATES_I]));
        }
      }
      atomicAdd(db + idx, float2type<FLASHRNN_DTYPE_B>(res));
    }
  }
}

} // anonymous namespace

namespace flashrnn_fused {

struct BackwardPassCut::private_data {
  int batch_size;
  int hidden_size;
  int num_heads;
  cublasHandle_t main_blas_handle;
  cudaStream_t main_stream;
  // event/stream/handle 0 is used for the inner loop,
  // others are used for outer mm's
  cublasHandle_t blas_handle_b[_NUM_BLAS_STREAMS];
  cudaStream_t stream_b[_NUM_BLAS_STREAMS];
  cudaEvent_t event_b[_NUM_BLAS_STREAMS];

  cudaStream_t stream_K;
  cudaEvent_t event_K;
};

BackwardPassCut::BackwardPassCut(const int batch_size, const int hidden_size,
                                 const int num_heads,
                                 const cublasHandle_t &blas_handle,
                                 const cudaStream_t &stream)
    : data_(new private_data) {
  data_->batch_size = batch_size;
  data_->hidden_size = hidden_size;
  data_->num_heads = num_heads;
  data_->main_blas_handle = blas_handle;
  data_->main_stream = stream;
  for (int i = 0; i < _NUM_BLAS_STREAMS; i++) {
    cublasCreate(&data_->blas_handle_b[i]);
    cudaStreamCreate(&data_->stream_b[i]);
    cudaEventCreateWithFlags(&data_->event_b[i], cudaEventDisableTiming);
  }

  cudaStreamCreate(&data_->stream_K);
  cudaEventCreateWithFlags(&data_->event_K, cudaEventDisableTiming);
}

BackwardPassCut::~BackwardPassCut() {
  for (int i = _NUM_BLAS_STREAMS - 1; i >= 0; i--) {
    cudaStreamSynchronize(data_->stream_b[i]);
    cublasDestroy(data_->blas_handle_b[i]);
    cudaEventDestroy(data_->event_b[i]);
    cudaStreamDestroy(data_->stream_b[i]);
  }
  cudaStreamSynchronize(data_->stream_K);
  cudaEventDestroy(data_->event_K);
  cudaStreamDestroy(data_->stream_K);
  delete data_;
}

void BackwardPassCut::Set(const int batch_size, const int hidden_size,
                          const int num_heads,
                          const cublasHandle_t &blas_handle,
                          const cudaStream_t &stream) {
  data_->batch_size = batch_size;
  data_->hidden_size = hidden_size;
  data_->main_blas_handle = blas_handle;
  data_->main_stream = stream;
}

int BackwardPassCut::Run(const int steps, const FLASHRNN_DTYPE_R *R_t,
                         const FLASHRNN_DTYPE_B *b, const FLASHRNN_DTYPE_S *s,
                         const FLASHRNN_DTYPE_S *ds_new, FLASHRNN_DTYPE_R *dR,
                         FLASHRNN_DTYPE_B *db, FLASHRNN_DTYPE_S *ds,
                         FLASHRNN_DTYPE_G *g_r, FLASHRNN_DTYPE_G *g_i,
                         FLASHRNN_DTYPE_G *g_bias,
                         FLASHRNN_ACC_DTYPE *d_state_buffer) { // [T*N]
  const DTYPE alpha = scalar_one<DTYPE>();
  const DTYPE beta_sum = scalar_one<DTYPE>(); // Accumulate into output matrix!
  const DTYPE beta_assign = scalar_zero<DTYPE>();
  const blas<void>::set_pointer_mode scoped1(data_->main_blas_handle);

  const int batch_size = data_->batch_size;
  const int hidden_size = data_->hidden_size;
  const int num_heads = data_->num_heads;
  const int head_dim = hidden_size / num_heads;
  const cublasHandle_t blas_handle = data_->main_blas_handle;
  const cudaStream_t stream = data_->main_stream;

  const cublasHandle_t *blas_handle_b = data_->blas_handle_b;
  const cudaStream_t *stream_b = data_->stream_b;
  const cudaEvent_t *event_b = data_->event_b;

  const cudaStream_t stream_K = data_->stream_K;
  const cudaEvent_t event_K = data_->event_K;

  // const int NH = batch_size * hidden_size;
  cudaStream_t save_stream;
  bool use_input_stream = false;
  if (cublasGetStream(blas_handle, &save_stream) == CUBLAS_STATUS_SUCCESS) {
    use_input_stream = true;
  } else {
    use_input_stream = false;
  }

  const uint recurrent_tiling_count_hidden = MIN(
      FLASHRNN_HIDDEN_SIZE / FLASHRNN_NUM_HEADS /
          FLASHRNN_BACKWARD_WARP_TILING_DIM_HIDDEN /
          FLASHRNN_BACKWARD_WARP_LOOPING_COUNT_HIDDEN, // because of minimal
                                                       // tiling dim tensorop
      MAX(WARP_SIZE * FLASHRNN_HIDDEN_SIZE / FLASHRNN_NUM_HEADS / 1024 /
              FLASHRNN_BACKWARD_WARP_TILING_DIM_HIDDEN,
          FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_HIDDEN)); // because the
                                                             // maximal block
                                                             // size is 1024

  if (recurrent_tiling_count_hidden !=
      FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_HIDDEN) {
    fprintf(stderr,
            "The specified BACKWARD RECURRENT_TILING_COUNT_HIDDEN "
            "should be: %d\n",
            recurrent_tiling_count_hidden);
    fprintf(stderr,
            "Values: RTCG: %d, RTCH: %d, WTCG: % d, BCB: "
            "% d, WTCH: % d\n ",
            FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_GATE,
            FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_HIDDEN,
            FLASHRNN_BACKWARD_WARP_TILING_COUNT_GATE,
            FLASHRNN_BACKWARD_BLOCK_TILING_COUNT_BATCH,
            FLASHRNN_BACKWARD_WARP_LOOPING_COUNT_HIDDEN);
    return 1;
  }

  const uint BLOCK_SIZE_HIDDEN = MIN(FLASHRNN_HIDDEN_SIZE, 512);
  const uint BLOCK_SIZE_BATCH = CEIL_DIV(512, BLOCK_SIZE_HIDDEN);

  const dim3 blockDim(BLOCK_SIZE_HIDDEN, BLOCK_SIZE_BATCH);

  // #pragma message(AT "Compiling Backward with BlockDim <" TOSTRING(              \
//     WARP_SIZE *FLASHRNN_BACKWARD_WARP_TILING_COUNT_GATE *FLASHRNN_HIDDEN_SIZE /      \
//     FLASHRNN_NUM_HEADS / FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_HIDDEN /           \
//     FLASHRNN_BACKWARD_WARP_LOOPING_COUNT_HIDDEN /                                 \
//     FLASHRNN_BACKWARD_WARP_TILING_DIM_HIDDEN) ", " TOSTRING(FLASHRNN_BACKWARD_WARP_TILING_COUNT_BATCH) ", 1>")

  const dim3 gridDim(CEIL_DIV(FLASHRNN_HIDDEN_SIZE, BLOCK_SIZE_HIDDEN),
                     CEIL_DIV(batch_size, BLOCK_SIZE_BATCH));

  FLASHRNNCellFusedBackwardCut<<<blockDim, gridDim, 0, stream_K>>>(
      steps, batch_size, R_t, b, s, g_r, g_i, g_bias, ds_new, ds);
  cudaEventRecord(event_b[0], stream);

  for (uint j = 0; j < _NUM_BLAS_STREAMS; j++) {
    cudaStreamWaitEvent(stream_b[j], event_K, 0);
  }

  cudaError_t err = cudaPeekAtLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Error after backward kernel launch: %s\n",
            cudaGetErrorString(err));
    return 1;
  }

  cublasSetStream(blas_handle_b[0], stream_b[0]);
  auto blas_res = blas<DTYPE>::gemmsb(
      blas_handle_b[0], CUBLAS_OP_N, CUBLAS_OP_T, head_dim,
      head_dim * FLASHRNN_NUM_GATES_R, batch_size * steps, &alpha, s,
      hidden_size, head_dim, g_r, hidden_size * FLASHRNN_NUM_GATES_R,
      head_dim * FLASHRNN_NUM_GATES_R, &beta_sum, dR, head_dim,
      FLASHRNN_NUM_GATES_R * head_dim * head_dim, num_heads);
  if (blas_res != CUBLAS_STATUS_SUCCESS) {
    return 1;
  }
  if (FLASHRNN_SIMPLE_AGG) {
    gradientBiasAggregationKernel<<<CEIL_DIV(FLASHRNN_NUM_GATES_T * hidden_size,
                                             512),
                                    512, 0, stream_b[1]>>>(
        hidden_size, batch_size, num_heads, steps, g_r, g_bias, db);
  } else {
    gradientBiasAggregationKernel<<<CEIL_DIV(FLASHRNN_NUM_GATES_T * hidden_size,
                                             512),
                                    512, 0, stream_b[1]>>>(
        hidden_size, batch_size, num_heads, steps, g_i, g_bias, db);
  }
  // TODO: add and test excessive biases
  err = cudaPeekAtLastError();
  if (err != cudaSuccess) {
    return 1;
  }

  for (uint j = 0; j < _NUM_BLAS_STREAMS; j++) {
    cudaEventRecord(event_b[j], stream_b[j]);
    if (use_input_stream) {
      cudaStreamWaitEvent(save_stream, event_b[j]);
    }
    cudaStreamWaitEvent(data_->main_stream, event_b[j]);
  }
  if (use_input_stream) {
    cublasSetStream(blas_handle, save_stream);
  }
  return 0;
}

} // namespace flashrnn_fused
