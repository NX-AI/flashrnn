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

#pragma once

#include "../util/util.h"

#ifndef FLASHRNN_NUM_GATES_R
// all needed definitions from external
#define FLASHRNN_NUM_HEADS 1
#define FLASHRNN_HIDDEN_SIZE 512
#define FLASHRNN_BATCH_SIZE 8
#define FLASHRNN_NUM_GATES_R 4
#define FLASHRNN_NUM_GATES_W 4
#define FLASHRNN_NUM_GATES_I 4
#define FLASHRNN_NUM_GATES_T 4
#define FLASHRNN_NUM_STATES 4
#define FLASHRNN_DTYPE __nv_bfloat16
#define FLASHRNN_USE_DTYPE_BFLOAT16
#define FLASHRNN_DTYPE_R __nv_bfloat16
#define FLASHRNN_DTYPE_B __nv_bfloat16
#define FLASHRNN_DTYPE_W __nv_bfloat16
#define FLASHRNN_DTYPE_G __nv_bfloat16
#define FLASHRNN_DTYPE_S __nv_bfloat16
#define FLASHRNN_DTYPE_A __nv_bfloat16

// fused forward
// optimized for hidden size 1024
#define FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN 1 // Rtch 16?
#define FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE 64  // Rtcg 1024 best 64
#define FLASHRNN_FORWARD_BLOCK_TILING_COUNT_BATCH 1      // Btcb
// means extra warps for threads
#define FLASHRNN_FORWARD_WARP_TILING_COUNT_BATCH 1 // Wtcb
// means each warp loops over batches stored in additional shared memory
#define FLASHRNN_FORWARD_WARP_LOOPING_COUNT_BATCH 1      // Wlcp
#define FLASHRNN_FORWARD_WARP_LOOPING_COUNT_GATE 1       // Wlcg
#define FLASHRNN_FORWARD_WARP_TILING_COUNT_HIDDEN 4      // Wtch 1024 best 8
#define FLASHRNN_FORWARD_WARP_RECURRENT_CACHED_HIDDEN 16 // Wrch 1024 best 8

#define FLASHRNN_FORWARD_MULTIHEAD_TILING_COUNT 1
#define FLASHRNN_FORWARD_SHARED_MEMORY_PADDING 8

#define FLASHRNN_FORWARD_WARP_TILING_DIM_BATCH 8   // Wtdb
#define FLASHRNN_FORWARD_WARP_TILING_DIM_HIDDEN 16 // Wtdg
#define FLASHRNN_FORWARD_WARP_TILING_DIM_GATE 32   // Wtdg

// fused backward
#define FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_HIDDEN 32 // Rtch 16?
#define FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_GATE 1    // Rtcg
#define FLASHRNN_BACKWARD_BLOCK_TILING_COUNT_BATCH 1       // Btcb
#define FLASHRNN_BACKWARD_WARP_TILING_COUNT_BATCH 1        // Wtcb
#define FLASHRNN_BACKWARD_WARP_LOOPING_COUNT_BATCH 1       // Wtlb
#define FLASHRNN_BACKWARD_WARP_LOOPING_COUNT_HIDDEN 1      // Wlch
#define FLASHRNN_BACKWARD_WARP_TILING_COUNT_GATE 8         // Wtcg
#define FLASHRNN_BACKWARD_WARP_RECURRENT_CACHED_GATE 32 // Wrcg optimal for 1024

#define FLASHRNN_BACKWARD_MULTIHEAD_TILING_COUNT 1
#define FLASHRNN_BACKWARD_SHARED_MEMORY_PADDING 8

#define FLASHRNN_BACKWARD_WARP_TILING_DIM_BATCH 8   // Wtdb
#define FLASHRNN_BACKWARD_WARP_TILING_DIM_GATE 16   // Wtdh
#define FLASHRNN_BACKWARD_WARP_TILING_DIM_HIDDEN 32 // Wtdh

// defines whether g = Wx + Ry + b for every gate, enables half the cache for
// backward
#define FLASHRNN_SIMPLE_AGG true
#endif

#ifdef FLASHRNN_USE_DTYPE_FLOAT32
#define FLASHRNN_ACC_DTYPE float
#endif
#ifdef FLASHRNN_USE_DTYPE_FLOAT16
#define FLASHRNN_ACC_DTYPE __half
#endif
#ifdef FLASHRNN_USE_DTYPE_BFLOAT16
#define FLASHRNN_ACC_DTYPE float
#endif

namespace flashrnn_fused {

class ForwardPass {
public:
  // training: `true` if the caller intends to perform a backward pass to
  // compute gradients. batch_size: the number of training/inference inputs
  // provided in each tensor. input_size: the dimension of each input vector.
  // hidden_size: the expected dimension of each output vector.
  // blas_handle: an initialized cuBLAS handle (see `cublasCreate`).
  ForwardPass(const bool training, const int batch_size, const int hidden_size,
              const int num_heads, const cublasHandle_t &blas_handle,
              const cudaStream_t &stream = 0);

  // Releases internal resources.
  // Blocks until all iterations have completed executing on the GPU.
  ~ForwardPass();

  // Set internal values for single forward / backward
  void Set(const bool training, const int batch_size, const int hidden_size,
           const int num_heads, const cublasHandle_t &blas_handle,
           const cudaStream_t &stream = 0);

  int Run(const int steps, const FLASHRNN_DTYPE_R *R, const FLASHRNN_DTYPE_B *b,
          const FLASHRNN_DTYPE_W *x, FLASHRNN_DTYPE_S *s, FLASHRNN_DTYPE_G *g_r,
          FLASHRNN_DTYPE_G *g_i, FLASHRNN_ACC_DTYPE *gate_buffer);

private:
  struct private_data;
  private_data *data_;
};

class BackwardPass {
public:
  // batch_size: the number of training inputs provided in each tensor.
  // input_size: the dimension of each input vector.
  // hidden_size: the expected dimension of each output vector.
  // blas_handle: an initialized cuBLAS handle (see `cublasCreate`).
  BackwardPass(const int batch_size, const int hidden_size, const int num_heads,
               const cublasHandle_t &blas_handle,
               const cudaStream_t &stream = 0);

  // Set internal values for single forward / backward
  void Set(const int batch_size, const int hidden_size, const int num_heads,
           const cublasHandle_t &blas_handle, const cudaStream_t &stream = 0);

  // Releases internal resources.
  // Blocks until all iterations have completed executing on the GPU.
  ~BackwardPass();

  int Run(const int steps, const FLASHRNN_DTYPE_R *R_t,
          const FLASHRNN_DTYPE_B *b, const FLASHRNN_DTYPE_S *s,
          const FLASHRNN_DTYPE_S *ds_new, FLASHRNN_DTYPE_R *dR,
          FLASHRNN_DTYPE_B *db, FLASHRNN_DTYPE_S *ds, FLASHRNN_DTYPE_G *g_r,
          FLASHRNN_DTYPE_G *g_i, FLASHRNN_DTYPE_G *g_b,
          FLASHRNN_ACC_DTYPE *d_state_buffer);

private:
  struct private_data;
  private_data *data_;
};

class BackwardPassCut {
public:
  // batch_size: the number of training inputs provided in each tensor.
  // input_size: the dimension of each input vector.
  // hidden_size: the expected dimension of each output vector.
  // blas_handle: an initialized cuBLAS handle (see `cublasCreate`).
  BackwardPassCut(const int batch_size, const int hidden_size,
                  const int num_heads, const cublasHandle_t &blas_handle,
                  const cudaStream_t &stream = 0);

  // Set internal values for single forward / backward
  void Set(const int batch_size, const int hidden_size, const int num_heads,
           const cublasHandle_t &blas_handle, const cudaStream_t &stream = 0);

  // Releases internal resources.
  // Blocks until all iterations have completed executing on the GPU.
  ~BackwardPassCut();

  int Run(const int steps, const FLASHRNN_DTYPE_R *R_t,
          const FLASHRNN_DTYPE_B *b, const FLASHRNN_DTYPE_S *s,
          const FLASHRNN_DTYPE_S *ds_new, FLASHRNN_DTYPE_R *dR,
          FLASHRNN_DTYPE_B *db, FLASHRNN_DTYPE_S *ds, FLASHRNN_DTYPE_G *g_r,
          FLASHRNN_DTYPE_G *g_i, FLASHRNN_DTYPE_G *g_bias,
          FLASHRNN_ACC_DTYPE *d_state_buffer);

private:
  struct private_data;
  private_data *data_;
};

} // namespace flashrnn_fused
