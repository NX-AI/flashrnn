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
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <vector>

#include "../util/blas.h"
#include "../util/cuda_error.h"
#include "../util/inline_ops.cuh"
#include "flashrnn.h"
#include "flashrnn_pointwise.cuh"

#define CEIL_DIV(a, b) (((a) + (b)-1) / (b))

#define MAX_THREADS_PER_BLOCK 512

namespace {
uint round_to_power2(uint x) {
  uint pow2 = 1;
  while (x > 1) {
    x >>= 1;
    pow2 <<= 1;
  }
  return pow2;
}

uint _min(uint a, uint b) { return (a > b) ? b : a; }
uint _max(uint a, uint b) { return (a < b) ? b : a; }

__global__ void gradientBiasAggregationKernel(
    const uint hidden_size, const uint batch_size, const uint num_heads,
    const uint steps, const uint num_gates_i, const uint num_gates_t,
    const FLASHRNN_DTYPE_G *gate_gradients_i,
    const FLASHRNN_DTYPE_G *gate_gradients_bias_only, FLASHRNN_DTYPE_B *db) {
  uint idx = threadIdx.x + blockDim.x * blockIdx.x;
  uint head_dim = hidden_size / num_heads;
  uint head_idx = idx / (head_dim * FLASHRNN_NUM_GATES_T);
  uint subhead_idx = idx % (head_dim * FLASHRNN_NUM_GATES_T);
  uint gate_idx = subhead_idx / head_dim;
  uint headint_idx = idx % head_dim;

  // if ((threadIdx.x == 0) && (blockIdx.x == 0)) {
  //   printf("Called kernel\n");
  // }

  if (idx < FLASHRNN_NUM_GATES_T * hidden_size) {

    if (gate_idx < FLASHRNN_NUM_GATES_I) {
      float res = 0.;
      for (uint t = 0; t < steps; t++) {
        for (uint b = 0; b < batch_size; b++) {
          res = add_g(
              res,
              type2float<FLASHRNN_DTYPE_G>(
                  gate_gradients_i[(t * batch_size + b) * hidden_size *
                                       FLASHRNN_NUM_GATES_I +
                                   FLASHRNN_NUM_GATES_I * head_dim * head_idx +
                                   gate_idx * head_dim + headint_idx]));
        }
      }
      // if (head_idx == 1) {
      //   printf("Hdidx: %d, %d, %d, %d, %d, %f\n", head_idx, subhead_idx,
      //          gate_idx, headint_idx, FLASHRNN_NUM_GATES_I, res);
      // }
      atomicAdd(db + head_idx * head_dim * FLASHRNN_NUM_GATES_T +
                    gate_idx * head_dim + headint_idx,
                float2type<FLASHRNN_DTYPE_B>(res));
    } else if (gate_idx < FLASHRNN_NUM_GATES_T) {
      float res = 0.;
      for (uint t = 0; t < steps; t++) {
        for (uint b = 0; b < batch_size; b++) {
          res = add_g(
              res, type2float<FLASHRNN_DTYPE_G>(
                       gate_gradients_bias_only
                           [((t * batch_size + b) * hidden_size +
                             head_dim * head_idx) *
                                (FLASHRNN_NUM_GATES_T - FLASHRNN_NUM_GATES_I) +
                            (gate_idx - FLASHRNN_NUM_GATES_I) * head_dim +
                            headint_idx]));
        }
      }
      atomicAdd(db + head_idx * head_dim * FLASHRNN_NUM_GATES_T +
                    gate_idx * head_dim + headint_idx,
                float2type<FLASHRNN_DTYPE_B>(res));
    }
  }
}
} // namespace

#define _NUM_BLAS_STREAMS 1

namespace flashrnn {

struct BackwardPassCut::private_data {
  int batch_size;
  int hidden_size;
  int num_heads;
  cublasHandle_t main_blas_handle;
  cudaStream_t main_stream;
  // event/stream/handle 0 is used for the inner loop, others are used for outer
  // mm's
  cudaStream_t stream_b[_NUM_BLAS_STREAMS];
  cudaEvent_t event_b[_NUM_BLAS_STREAMS];
  cublasHandle_t blas_handle_R;
  cudaStream_t stream_R;
  cudaEvent_t event_R;
  cudaDeviceProp deviceProperties;
};

BackwardPassCut::BackwardPassCut(const int batch_size, const int hidden_size,
                                 const int num_heads,
                                 const cublasHandle_t &blas_handle,
                                 const cudaStream_t &stream)
    : data_(new private_data) {
  const FLASHRNN_DTYPE_R alpha = scalar_one<FLASHRNN_DTYPE_R>();
  data_->batch_size = batch_size;
  data_->hidden_size = hidden_size;
  data_->num_heads = num_heads;
  data_->main_blas_handle = blas_handle;
  data_->main_stream = stream;

  for (int i = 0; i < _NUM_BLAS_STREAMS; i++) {
    cudaStreamCreate(&data_->stream_b[i]);
    cudaEventCreateWithFlags(&data_->event_b[i], cudaEventDisableTiming);
  }
  cublasCreate(&data_->blas_handle_R);
  cudaStreamCreate(&data_->stream_R);
  cudaEventCreateWithFlags(&data_->event_R, cudaEventDisableTiming);
  cublasSetStream(data_->blas_handle_R, data_->stream_R);
}

BackwardPassCut::~BackwardPassCut() {
  for (int i = 0; i < _NUM_BLAS_STREAMS; i++) {
    cudaStreamSynchronize(data_->stream_b[i]);
    cudaEventDestroy(data_->event_b[i]);
    cudaStreamDestroy(data_->stream_b[i]);
  }
  cudaStreamSynchronize(data_->stream_R);
  cublasDestroy(data_->blas_handle_R);
  cudaEventDestroy(data_->event_R);
  cudaStreamDestroy(data_->stream_R);
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

int BackwardPassCut::Iterate(const cudaStream_t &stream,
                             const FLASHRNN_DTYPE_R *R_t,    // [H*4,H]
                             const FLASHRNN_DTYPE_B *b,      // [H*4]
                             const FLASHRNN_DTYPE_S *s,      // [N,H]
                             const FLASHRNN_DTYPE_S *s_new,  // [N,H]
                             const FLASHRNN_DTYPE_S *ds_new, // [N,H]
                             FLASHRNN_DTYPE_R *dR,           // [H,H*4]
                             FLASHRNN_DTYPE_B *db,           // [H*4]
                             FLASHRNN_DTYPE_S *ds,           // [N,H]
                             FLASHRNN_DTYPE_G *g_r,          // [N,H*4]
                             FLASHRNN_DTYPE_G *g_i,
                             FLASHRNN_DTYPE_G *g_b) { // [N]

  const FLASHRNN_DTYPE_R alpha = scalar_one<FLASHRNN_DTYPE_R>();
  const FLASHRNN_DTYPE_R beta_sum =
      scalar_one<FLASHRNN_DTYPE_R>(); // Accumulate into
                                      // output matrix!
  const FLASHRNN_DTYPE_R beta_assign = scalar_zero<FLASHRNN_DTYPE_R>();

  const blas<void>::set_pointer_mode scoped1(data_->main_blas_handle);
  int res = 0;
  const int batch_size = data_->batch_size;
  const int hidden_size = data_->hidden_size;
  const int num_heads = data_->num_heads;
  const uint head_dim = hidden_size / num_heads;
  const cublasHandle_t blas_handle = data_->main_blas_handle;

  cudaStream_t *stream_b = data_->stream_b;
  cudaStream_t stream_R = data_->stream_R;
  const cublasHandle_t blas_handle_R = data_->blas_handle_R;

  cudaEvent_t event_R = data_->event_R;
  cudaEvent_t *event_b = data_->event_b;
  // cudaEvent_t event5 = data_->event[_NUM_BLAS_STREAMS];

  const cudaEvent_t event = data_->event_b[0];

  cudaStream_t save_stream;
  cublasGetStream(blas_handle, &save_stream);

  // Make sure inputs are ready before using them.
  if (stream) {
    cudaEventRecord(event, stream);
    cudaStreamWaitEvent(stream_R, event, 0);
  }
  const uint BH = batch_size * hidden_size;
  IterateInternal(R_t, b, s, BH, s_new, BH, ds_new, BH, ds, BH, g_r, g_i, g_b);

  // Wait for pointwise operations to complete since there's a
  // data dependency between its output (`v`) and the following matmuls.
  for (uint i = 0; i < _NUM_BLAS_STREAMS; i++) {
    cudaStreamWaitEvent(stream_b[i], event_R, 0);
  }

  cublasSetStream(blas_handle_R, stream_R);
  blas<FLASHRNN_DTYPE_R>::gemmsb(
      blas_handle_R, CUBLAS_OP_N, CUBLAS_OP_T, head_dim * FLASHRNN_NUM_GATES_R,
      head_dim, batch_size, &alpha, g_r, hidden_size * FLASHRNN_NUM_GATES_R,
      head_dim * FLASHRNN_NUM_GATES_R, s, hidden_size, head_dim, &beta_sum, dR,
      head_dim * FLASHRNN_NUM_GATES_R,
      head_dim * head_dim * FLASHRNN_NUM_GATES_R, num_heads);
  cudaEventRecord(event_R, stream_R);
  cudaStreamWaitEvent(save_stream, event_R);

  cudaStreamWaitEvent(stream_b[0], event, 0);
  if (FLASHRNN_SIMPLE_AGG) {
    gradientBiasAggregationKernel<<<CEIL_DIV(FLASHRNN_NUM_GATES_T * hidden_size,
                                             512),
                                    512, 0, stream_b[0]>>>(
        hidden_size, batch_size, num_heads, 1, FLASHRNN_NUM_GATES_I,
        FLASHRNN_NUM_GATES_T, g_r, g_b, db);
  } else {
    gradientBiasAggregationKernel<<<CEIL_DIV(FLASHRNN_NUM_GATES_T * hidden_size,
                                             512),
                                    512, 0, stream_b[0]>>>(
        hidden_size, batch_size, num_heads, 1, FLASHRNN_NUM_GATES_I,
        FLASHRNN_NUM_GATES_T, g_i, g_b, db);
  }
  auto cuda_res = cudaPeekAtLastError();
  if (cuda_res != cudaSuccess) {
    res = 1;
  }
  cudaEventRecord(event_b[0], stream_b[0]);

  for (uint i = 0; i < _NUM_BLAS_STREAMS; i++) {
    cudaStreamWaitEvent(save_stream, event_b[i]);
  }
  cudaStreamWaitEvent(save_stream, event_R);
  cublasSetStream(blas_handle, save_stream);
  return res;
}

int BackwardPassCut::IterateInternal(const FLASHRNN_DTYPE_R *R_t, // [H*4,H]
                                     const FLASHRNN_DTYPE_B *b,
                                     const FLASHRNN_DTYPE_S *s, // [N,H]
                                     const uint s_stride,
                                     const FLASHRNN_DTYPE_S *s_new, // [N,H]
                                     const uint s_new_stride,
                                     const FLASHRNN_DTYPE_S *ds_new, // [N,H]
                                     const uint ds_new_stride,
                                     FLASHRNN_DTYPE_S *ds, // [N,H]
                                     const uint ds_stride,
                                     FLASHRNN_DTYPE_G *g_r,
                                     FLASHRNN_DTYPE_G *g_i,
                                     FLASHRNN_DTYPE_G *g_b) { // [N,H*4]

  const FLASHRNN_DTYPE_R alpha = scalar_one<FLASHRNN_DTYPE_R>();
  const FLASHRNN_DTYPE_R beta_sum =
      scalar_one<FLASHRNN_DTYPE_R>(); // Accumulate into
                                      // output matrix!

  const int batch_size = data_->batch_size;
  const int hidden_size = data_->hidden_size;
  const int num_heads = data_->num_heads;
  const int head_dim = hidden_size / num_heads;
  const cudaStream_t stream_R = data_->stream_R;
  const cudaEvent_t event_R = data_->event_R;

  int res = 0;

  // Compute launch configuration for pointwise operations kernel.
  uint gridDimHead =
      _min(_max(32, round_to_power2(head_dim)), MAX_THREADS_PER_BLOCK);
  uint gridDimBatch = _max(1, _min(MAX_THREADS_PER_BLOCK / gridDimHead,
                                   round_to_power2(batch_size)));

  const dim3 blockDim(gridDimHead, gridDimBatch, 1);

  const dim3 gridDim((head_dim + blockDim.x - 1) / blockDim.x,
                     (batch_size + blockDim.y - 1) / blockDim.y, num_heads);
  FLASHRNNPointwiseBackward<<<gridDim, blockDim, 0, stream_R>>>(
      batch_size, hidden_size, num_heads, s, s_stride, g_r, g_i, b, s_new,
      s_new_stride, ds_new, ds_new_stride, ds, ds_stride, g_r, g_i, g_b);
  auto cuda_res = cudaPeekAtLastError();
  if (cuda_res != cudaSuccess) {
    res = 1;
  }

  cudaEventRecord(event_R, stream_R);

  return res;
}

int BackwardPassCut::Run(const int steps, const FLASHRNN_DTYPE_R *R_t,
                         const FLASHRNN_DTYPE_B *b, const FLASHRNN_DTYPE_S *s,
                         const FLASHRNN_DTYPE_S *ds_new, FLASHRNN_DTYPE_R *dR,
                         FLASHRNN_DTYPE_B *db, FLASHRNN_DTYPE_S *ds,
                         FLASHRNN_DTYPE_G *g_r, FLASHRNN_DTYPE_G *g_i,
                         FLASHRNN_DTYPE_G *g_bias) {
  const FLASHRNN_DTYPE_R alpha = scalar_one<FLASHRNN_DTYPE_R>();
  const FLASHRNN_DTYPE_R beta_sum =
      scalar_one<FLASHRNN_DTYPE_R>(); // Accumulate into
                                      // output matrix!
  const FLASHRNN_DTYPE_R beta_assign = scalar_zero<FLASHRNN_DTYPE_R>();
  const blas<void>::set_pointer_mode scoped1(data_->main_blas_handle);

  const int batch_size = data_->batch_size;
  const int hidden_size = data_->hidden_size;
  const int num_heads = data_->num_heads;
  const int head_dim = hidden_size / num_heads;
  const cublasHandle_t blas_handle = data_->main_blas_handle;
  const cudaEvent_t event_R = data_->event_R;
  const cudaEvent_t *event_b = data_->event_b;
  const int BH = batch_size * hidden_size;
  cudaStream_t save_stream;
  cudaStream_t *stream_b = data_->stream_b;
  cudaStream_t stream_R = data_->stream_R;
  const cublasHandle_t blas_handle_R = data_->blas_handle_R;
  bool use_input_stream = false;
  int res = 0;

  if (cublasGetStream(blas_handle, &save_stream) == CUBLAS_STATUS_SUCCESS) {
    use_input_stream = true;
  } else {
    use_input_stream = false;
  }
  const uint state_stride = (steps + 1) * BH;
  cublasSetStream(blas_handle_R, stream_R);

  for (int t = steps - 1; t >= 0; --t) {
    res |= IterateInternal(
        R_t, b, s + t * BH, state_stride, s + (t + 1) * BH, state_stride,
        ds_new + (t + 1) * BH, state_stride, ds, BH,
        g_r + t * BH * FLASHRNN_NUM_GATES_R,
        g_i + t * BH * FLASHRNN_NUM_GATES_I,
        g_bias + t * BH * (FLASHRNN_NUM_GATES_T - FLASHRNN_NUM_GATES_I));
  }
  if (res != 0) {
    fprintf(stderr, "Error during alternating kernels.\n");
  }

  for (uint i = 0; i < _NUM_BLAS_STREAMS; i++) {
    cudaStreamWaitEvent(stream_b[i], event_R, 0);
  }

  auto blas_err = blas<FLASHRNN_DTYPE_R>::gemmsb(
      blas_handle_R, CUBLAS_OP_N, CUBLAS_OP_T, head_dim * FLASHRNN_NUM_GATES_R,
      head_dim, batch_size * steps, &alpha, g_r,
      hidden_size * FLASHRNN_NUM_GATES_R, head_dim * FLASHRNN_NUM_GATES_R, s,
      hidden_size, head_dim, &beta_sum, dR, head_dim * FLASHRNN_NUM_GATES_R,
      head_dim * head_dim * FLASHRNN_NUM_GATES_R, num_heads);
  res |= (blas_err == CUBLAS_STATUS_SUCCESS) ? 0 : 1;
  if (res != 0) {
    fprintf(stderr, "Error during dR matmul.\n");
  }

  if (FLASHRNN_SIMPLE_AGG) {
    gradientBiasAggregationKernel<<<CEIL_DIV(FLASHRNN_NUM_GATES_T * hidden_size,
                                             512),
                                    512, 0, stream_b[0]>>>(
        hidden_size, batch_size, num_heads, steps, FLASHRNN_NUM_GATES_I,
        FLASHRNN_NUM_GATES_T, g_r, g_bias, db);
  } else {
    gradientBiasAggregationKernel<<<CEIL_DIV(FLASHRNN_NUM_GATES_T * hidden_size,
                                             512),
                                    512, 0, stream_b[0]>>>(
        hidden_size, batch_size, num_heads, steps, FLASHRNN_NUM_GATES_I,
        FLASHRNN_NUM_GATES_T, g_i, g_bias, db);
  }
  auto cuda_res = cudaPeekAtLastError();
  if (cuda_res != cudaSuccess) {
    res = 1;
  }
  if (res != 0) {
    fprintf(stderr, "Error during bias gradient kernel.\n");
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
  return res;
}

} // namespace flashrnn
