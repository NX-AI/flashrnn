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

#ifndef _FLASHRNN_POINTWISE_INCLUDED
#include "flashrnn_fused_pointwise_base.cuh"
#endif

#include "flashrnn.h"
#include <cooperative_groups.h>
#include <driver_types.h>
#include <mma.h>
#include <stdio.h>

#define CEIL_DIV(a, b) (((a) + (b)-1) / (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

// #define DEBUG

namespace {

namespace cg = cooperative_groups;
using namespace nvcuda;

// optimal values for hidden size 1024 on A100
#ifndef FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_GATE

#define FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_HIDDEN 32 // BRTCH 16?
#define FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_GATE 1    // BRTCG
#define FLASHRNN_BACKWARD_BLOCK_TILING_COUNT_BATCH 1       // Btcb
#define FLASHRNN_BACKWARD_WARP_TILING_COUNT_BATCH 1        // Wtcb
#define FLASHRNN_BACKWARD_WARP_LOOPING_COUNT_BATCH 1       // Wtlb
#define FLASHRNN_BACKWARD_WARP_LOOPING_COUNT_HIDDEN 1      // BWLCH
#define FLASHRNN_BACKWARD_WARP_TILING_COUNT_GATE 8         // BWTCG
#define FLASHRNN_BACKWARD_WARP_RECURRENT_CACHED_GATE                           \
  32 // BWRCG optimal for 1024

#define FLASHRNN_BACKWARD_MULTIHEAD_TILING_COUNT 1
// offset to reduce memory bank conflicts in shared memory
#define FLASHRNN_BACKWARD_SHARED_MEMORY_PADDING 8

#define FLASHRNN_HIDDEN_SIZE 1024
#define FLASHRNN_NUM_HEADS 1

#define FLASHRNN_BACKWARD_WARP_TILING_DIM_BATCH 8   // BWTDB
#define FLASHRNN_BACKWARD_WARP_TILING_DIM_HIDDEN 32 // BWTDH

#endif

#define BRTCH FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_HIDDEN
#define BRTCG FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_GATE
#define BBTCB FLASHRNN_BACKWARD_BLOCK_TILING_COUNT_BATCH
#define BWTCB FLASHRNN_BACKWARD_WARP_TILING_COUNT_BATCH
#define BWLCB FLASHRNN_BACKWARD_WARP_LOOPING_COUNT_BATCH
#define BWLCH FLASHRNN_BACKWARD_WARP_LOOPING_COUNT_HIDDEN
#define BWTCG FLASHRNN_BACKWARD_WARP_TILING_COUNT_GATE
#define BWRCG FLASHRNN_BACKWARD_WARP_RECURRENT_CACHED_GATE
#define BMHTC FLASHRNN_BACKWARD_MULTIHEAD_TILING_COUNT
#define BSMP FLASHRNN_BACKWARD_SHARED_MEMORY_PADDING
#define BWTDB FLASHRNN_BACKWARD_WARP_TILING_DIM_BATCH
#define BWTDG FLASHRNN_BACKWARD_WARP_TILING_DIM_GATE
#define BWTDH FLASHRNN_BACKWARD_WARP_TILING_DIM_HIDDEN

#ifdef FLASHRNN_USE_DTYPE_FLOAT32
#define DTYPE float
#define MAT_DTYPE wmma::precision::tf32
#define ACC_DTYPE float
#endif
#ifdef FLASHRNN_USE_DTYPE_FLOAT16
#define DTYPE __half
#define MAT_DTYPE __half
#define ACC_DTYPE __half
#endif
#ifdef FLASHRNN_USE_DTYPE_BFLOAT16
#define DTYPE __nv_bfloat16
#define MAT_DTYPE __nv_bfloat16
#define ACC_DTYPE float
#endif

#define HS FLASHRNN_HIDDEN_SIZE
#define NH FLASHRNN_NUM_HEADS

#define WARP_SIZE 32 // warpSize = 32 threads

#define _NUM_BLAS_STREAMS 2
#define _FLOAT4FACTOR sizeof(float4) / sizeof(ACC_DTYPE)

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
    FLASHRNNCellFusedBackward(
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
        const FLASHRNN_DTYPE_S *d_states_ext, FLASHRNN_DTYPE_S *d_states,
        ACC_DTYPE *d_state_buffer) {
  const uint head_dim = FLASHRNN_HIDDEN_SIZE / FLASHRNN_NUM_HEADS;

  const uint gate_grid_dim = BRTCG; // == gridDim.z / BMHTC
  if (gate_grid_dim != gridDim.z / BMHTC) {
    printf("Bad gate grid Dim, this should not be possible!\n");
  }

  const uint gate_block_idx = blockIdx.z % gate_grid_dim;
  const uint multihead_idx = blockIdx.z / gate_grid_dim * head_dim;
  // assuming at least 8 as a batch size, at least 32 as a
  // hidden dim this is necessary for mm each thread takes
  // a tile of (8 x 32) of the pre-activations of one gate,
  // i.e. a (8 x 32) tile of the outputs

  /// tile of R within hidden_size / Rtdh, FLASHRNN_NUM_GATES_R *
  /// hidden_size / Rtdg
  extern __shared__ float4 sbuf[];
  FLASHRNN_DTYPE_R *R_shared = (FLASHRNN_DTYPE_R *)sbuf;

  int gate_dim_per_block_shared =
      (FLASHRNN_NUM_GATES_R * head_dim / BRTCG - BWTDG * BWTCG * BWRCG);
  if (gate_dim_per_block_shared < 0) {
    gate_dim_per_block_shared = 0;
  }
  // matrix multiplication buffer of size (batch_dim x
  // hidden_size / Rtdg)
  ACC_DTYPE *mmul_buffer =
      (ACC_DTYPE *)(((FLASHRNN_DTYPE_R *)(sbuf)) +
                    (head_dim / BRTCH) * (gate_dim_per_block_shared + BSMP));
  if (gate_dim_per_block_shared == 0) {
    mmul_buffer = (ACC_DTYPE *)(sbuf);
  }

  const uint BatchIterations =
      batch_dim / BWLCB / BWTDB / gridDim.y / blockDim.y;
  const uint batch_idx =
      BWLCB * BWTDB * (blockIdx.y * blockDim.y + threadIdx.y);
  const uint block_batch_idx = BWLCB * BWTDB * threadIdx.y;
  const uint state_warp_idx = BWTDH * BWLCH *
                              ((blockDim.x / BWTCG * blockIdx.x +
                                (threadIdx.x % (blockDim.x / BWTCG))) /
                               warpSize);
  const uint state_warp_local_idx =
      BWTDH * BWLCH * ((threadIdx.x % (blockDim.x / BWTCG)) / warpSize);
  const uint state_blocklevel_idx = state_warp_local_idx + threadIdx.x % BWTDH;
  const uint state_warp_overcount = warpSize / BWTDH;

  const uint wtcg_idx = threadIdx.x / (blockDim.x / BWTCG);

  FLASHRNN_DTYPE_S ds_inout_local[CEIL_DIV(BWLCB * BWTDB * BWLCH * BWTDH,
                                           BRTCG * BWTCG * WARP_SIZE)]
                                 [FLASHRNN_NUM_STATES];
#if FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_GATE *                            \
        FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_HIDDEN >                      \
    1
  auto gr = cg::this_grid();

#endif

#if FLASHRNN_BACKWARD_WARP_RECURRENT_CACHED_GATE > 0
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, BWTDB, BWTDH, BWTDG, MAT_DTYPE,
                         nvcuda::wmma::col_major>
      b_frag_cache[BWLCH][BWRCG];

  // store R to registers
  if (BWRCG > 0) {
    for (uint wlch_idx = 0; wlch_idx < BWLCH; wlch_idx++) {
      for (uint wrcg_idx = 0; wrcg_idx < BWRCG; wrcg_idx++) {
        const uint R_offset =
            FLASHRNN_NUM_GATES_R * multihead_idx * head_dim +
            (blockIdx.x * head_dim / BRTCH + state_warp_local_idx / BWLCH +
             wlch_idx * BWTDH) *
                FLASHRNN_NUM_GATES_R * head_dim +
            gate_block_idx * (FLASHRNN_NUM_GATES_R * head_dim / BRTCG) +
            wtcg_idx * (FLASHRNN_NUM_GATES_R * head_dim / BRTCG / BWTCG) +
            wrcg_idx * BWTDG;
        nvcuda::wmma::load_matrix_sync(b_frag_cache[wlch_idx][wrcg_idx],
                                       R + R_offset,
                                       FLASHRNN_NUM_GATES_R * head_dim);
      }
    }
  }
#endif

  // move R via float4's
  const uint local_gate_offset = BWRCG * BWTDG;
  const uint local_gate_warp_dim =
      FLASHRNN_NUM_GATES_R * head_dim / BRTCG / BWTCG;

  if (gate_dim_per_block_shared > 0) {

    for (uint j = 0; j < (gate_dim_per_block_shared)*head_dim / BRTCH;
         j += blockDim.y * blockDim.x) {
      const uint local_linear_idx =
          (j + threadIdx.y * blockDim.x + threadIdx.x);
      const uint local_hidden_idx =
          local_linear_idx / (gate_dim_per_block_shared);
      const uint local_gate_idx =
          (local_linear_idx) % (gate_dim_per_block_shared);
      const uint local_gate_warp_idx =
          local_gate_idx % MAX(1, local_gate_warp_dim - local_gate_offset);
      const uint local_gate_block_idx =
          local_gate_idx / MAX(1, local_gate_warp_dim - local_gate_offset);
      const uint global_idx =
          FLASHRNN_NUM_GATES_R * multihead_idx * head_dim +
          (blockIdx.x * head_dim / BRTCH + local_hidden_idx) *
              FLASHRNN_NUM_GATES_R * head_dim + // hidden
          gate_block_idx * (FLASHRNN_NUM_GATES_R * head_dim / BRTCG) +
          local_gate_offset + local_gate_warp_idx +
          local_gate_block_idx * local_gate_warp_dim; // hidden

      if (local_gate_idx < FLASHRNN_NUM_GATES_R * head_dim / BRTCG) {
        ((R_shared + local_hidden_idx * (gate_dim_per_block_shared + BSMP) +
          local_gate_idx))[0] = ((R + global_idx))[0];
      }
    }
  }

  __syncthreads();

  // fragments for matrix multiplication
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, BWTDB, BWTDH, BWTDG, MAT_DTYPE,
                         nvcuda::wmma::row_major>
      a_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, BWTDB, BWTDH, BWTDG, MAT_DTYPE,
                         nvcuda::wmma::col_major>
      b_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, BWTDB, BWTDH, BWTDG,
                         ACC_DTYPE>
      c_frag[BWLCB];

  for (uint batch_it = 0; batch_it < BatchIterations; batch_it++) {
    uint rgate_offset =
        FLASHRNN_NUM_GATES_R * FLASHRNN_HIDDEN_SIZE * batch_dim * (steps - 1);
    uint igate_offset =
        FLASHRNN_NUM_GATES_I * FLASHRNN_HIDDEN_SIZE * batch_dim * (steps - 1);
    uint bgate_offset = (FLASHRNN_NUM_GATES_T - FLASHRNN_NUM_GATES_I) *
                        FLASHRNN_HIDDEN_SIZE * batch_dim * (steps - 1);
    uint state_offset = (steps - 1) * FLASHRNN_HIDDEN_SIZE * batch_dim;

    // load ds_inout_local from ds_inout
    const uint overcount = MAX(1, blockDim.x / (head_dim / BRTCH / BWLCH));
    const uint overcount_idx = threadIdx.x / (head_dim / BRTCH / BWLCH);
#pragma unroll
    for (uint local_it = 0; local_it < CEIL_DIV(BWLCB * BWTDB * BWLCH * BWTDH,
                                                BRTCG * BWTCG * WARP_SIZE);
         local_it++) {
      const uint local_total_it = local_it * gate_grid_dim * overcount +
                                  gate_block_idx * overcount + overcount_idx;

      const uint wlch_idx = local_total_it % BWLCH;
      const uint local_batch_idx = local_total_it / BWLCH;
      if (local_batch_idx < BWLCB * BWTDB) {
        const uint local_state_idx = threadIdx.x % (head_dim / BRTCH / BWLCH) +
                                     wlch_idx * (head_dim / BRTCH / BWLCH);
        const uint global_state_idx = multihead_idx +
                                      (state_warp_idx - state_warp_local_idx) +
                                      local_state_idx;
        const uint global_batch_idx =
            batch_idx + batch_it * blockDim.y * gridDim.y * BWLCB * BWTDB +
            local_batch_idx;
        for (uint sub_state_idx = 0; sub_state_idx < FLASHRNN_NUM_STATES;
             sub_state_idx++) {
          ds_inout_local[local_it][sub_state_idx] =
              d_states[FLASHRNN_HIDDEN_SIZE * global_batch_idx +
                       global_state_idx +
                       sub_state_idx * FLASHRNN_HIDDEN_SIZE * batch_dim];
        }
      }
    }

    for (uint t = steps; t >= 1; t--) {

      // pointwise ops
#pragma unroll
      for (uint local_it = 0; local_it < CEIL_DIV(BWLCB * BWTDB * BWLCH * BWTDH,
                                                  BRTCG * BWTCG * WARP_SIZE);
           local_it++) {
        const uint local_total_it = local_it * gate_grid_dim * overcount +
                                    gate_block_idx * overcount + overcount_idx;

        const uint wlch_idx = local_total_it % BWLCH;
        const uint local_batch_idx = local_total_it / BWLCH;
        const uint local_state_idx = threadIdx.x % (head_dim / BRTCH / BWLCH) +
                                     wlch_idx * (head_dim / BRTCH / BWLCH);
        const uint global_state_idx = multihead_idx +
                                      (state_warp_idx - state_warp_local_idx) +
                                      local_state_idx;

        const uint global_batch_idx =
            batch_idx + batch_it * blockDim.y * gridDim.y * BWLCB * BWTDB +
            local_batch_idx;
        if (local_batch_idx < BWLCB * BWTDB) {
          FLASHRNNPointwiseBackward(
              g_r_inout + rgate_offset +
                  FLASHRNN_NUM_GATES_R * FLASHRNN_HIDDEN_SIZE *
                      global_batch_idx +
                  FLASHRNN_NUM_GATES_R * global_state_idx,
              1,
              g_i_inout + igate_offset +
                  FLASHRNN_NUM_GATES_I * FLASHRNN_HIDDEN_SIZE *
                      global_batch_idx +
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
              b + global_state_idx * FLASHRNN_NUM_GATES_T +
                  FLASHRNN_NUM_GATES_I,
              ds_inout_local[local_it], 1,
              g_r_inout + rgate_offset +
                  FLASHRNN_NUM_GATES_R * FLASHRNN_HIDDEN_SIZE *
                      global_batch_idx +
                  FLASHRNN_NUM_GATES_R * global_state_idx,
              1,
              g_i_inout + igate_offset +
                  FLASHRNN_NUM_GATES_I * FLASHRNN_HIDDEN_SIZE *
                      global_batch_idx +
                  FLASHRNN_NUM_GATES_I * global_state_idx,
              1,
              g_b + bgate_offset +
                  (FLASHRNN_NUM_GATES_T - FLASHRNN_NUM_GATES_I) *
                      (FLASHRNN_HIDDEN_SIZE * global_batch_idx +
                       global_state_idx),
              1);
        }
      }

#if FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_GATE *                            \
        FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_HIDDEN >                      \
    1
      gr.sync();
#else
      __syncthreads();
#endif

      // matmul for d_states
      for (uint wlch_idx = 0; wlch_idx < BWLCH; wlch_idx++) {

        if (state_warp_local_idx + wlch_idx * BWTDH < head_dim / BRTCH) {

          for (uint local_batch_idx = 0; local_batch_idx < BWLCB;
               local_batch_idx++) {
            // Initialize the output to zero
            nvcuda::wmma::fill_fragment(c_frag[local_batch_idx], 0.0f);
          }
          // accumulating matrix multiplications
#if FLASHRNN_BACKWARD_WARP_RECURRENT_CACHED_GATE > 0
#pragma unroll
          for (uint wrcg_idx = 0; wrcg_idx < BWRCG; wrcg_idx++) {
            const uint midx =
                wtcg_idx * (FLASHRNN_NUM_GATES_R * head_dim / BRTCG / BWTCG) +
                wrcg_idx * BWTDG;
            // Load the inputs
            for (uint local_batch_idx = 0; local_batch_idx < BWLCB;
                 local_batch_idx++) {
              nvcuda::wmma::load_matrix_sync(
                  a_frag,
                  g_r_inout + rgate_offset +
                      ((batch_idx +
                        batch_it * BWTDB * BWLCB * blockDim.y * gridDim.y +
                        BWTDB * local_batch_idx) *
                       FLASHRNN_NUM_GATES_R * FLASHRNN_HIDDEN_SIZE) +
                      FLASHRNN_NUM_GATES_R * multihead_idx +
                      gate_block_idx *
                          (FLASHRNN_NUM_GATES_R * head_dim / BRTCG) +
                      midx,
                  FLASHRNN_NUM_GATES_R * FLASHRNN_HIDDEN_SIZE);
              nvcuda::wmma::mma_sync(c_frag[local_batch_idx], a_frag,
                                     b_frag_cache[wlch_idx][wrcg_idx],
                                     c_frag[local_batch_idx]);
            }
          }
#endif
          const uint R_offset = (state_warp_local_idx + wlch_idx * BWTDH) *
                                (gate_dim_per_block_shared + BSMP);

          // accumulating matrix multiplications
          for (uint midx = wtcg_idx * (FLASHRNN_NUM_GATES_R * head_dim / BRTCG /
                                       BWTCG) +
                           BWRCG * BWTDG;
               midx <
               (wtcg_idx + 1) * FLASHRNN_NUM_GATES_R * head_dim / BRTCG / BWTCG;
               midx += BWTDG) {
            // Load the inputs
            const uint R_idx =
                R_offset +
                (midx -
                 (wtcg_idx * (FLASHRNN_NUM_GATES_R * head_dim / BRTCG / BWTCG) +
                  BWRCG * BWTDG) +
                 wtcg_idx * (FLASHRNN_NUM_GATES_R * head_dim / BRTCG / BWTCG -
                             BWRCG * BWTDG));
            nvcuda::wmma::load_matrix_sync(b_frag, R_shared + R_idx,
                                           (gate_dim_per_block_shared + BSMP));
#pragma unroll
            for (uint local_batch_idx = 0; local_batch_idx < BWLCB;
                 local_batch_idx++) {
              nvcuda::wmma::load_matrix_sync(
                  a_frag,
                  g_r_inout + rgate_offset +
                      ((batch_idx +
                        batch_it * BWTDB * BWLCB * blockDim.y * gridDim.y +
                        BWTDB * local_batch_idx) *
                       FLASHRNN_NUM_GATES_R * FLASHRNN_HIDDEN_SIZE) +
                      FLASHRNN_NUM_GATES_R * multihead_idx +
                      gate_block_idx *
                          (FLASHRNN_NUM_GATES_R * head_dim / BRTCG) +
                      midx,
                  FLASHRNN_NUM_GATES_R * FLASHRNN_HIDDEN_SIZE);
              nvcuda::wmma::mma_sync(c_frag[local_batch_idx], a_frag, b_frag,
                                     c_frag[local_batch_idx]);
            }
          }
#pragma unroll
          for (uint local_batch_idx = 0; local_batch_idx < BWLCB;
               local_batch_idx++) {
            nvcuda::wmma::store_matrix_sync(
                mmul_buffer +
                    (local_batch_idx * BWTDB +
                     wtcg_idx * (blockDim.y * BWTDB * BWLCB) +
                     block_batch_idx) *
                        (head_dim / BRTCH + BSMP) +
                    state_warp_local_idx + wlch_idx * BWTDH,
                c_frag[local_batch_idx], (head_dim / BRTCH + BSMP),
                nvcuda::wmma::mem_row_major);
          }

          // // accumulate in BWLCH dimension
          if (BWTCG > 1) {
            __syncthreads();

            // accumulate along BWLCH tiling dimension
            for (uint local_batch_idx = wtcg_idx * state_warp_overcount +
                                        (threadIdx.x % warpSize) / BWTDH;
                 local_batch_idx < BWLCB * BWTDB;
                 local_batch_idx += BWTCG * state_warp_overcount) {
              for (uint local_wtcg_idx = 1; local_wtcg_idx < BWTCG;
                   local_wtcg_idx++) {
                mmul_buffer[(local_batch_idx + block_batch_idx) *
                                (head_dim / BRTCH + BSMP) +
                            state_blocklevel_idx + wlch_idx * BWTDH] =
                    add_g(
                        mmul_buffer[(local_batch_idx + block_batch_idx) *
                                        (head_dim / BRTCH + BSMP) +
                                    state_blocklevel_idx + wlch_idx * BWTDH],
                        mmul_buffer[local_wtcg_idx * BWLCB * BWTDB *
                                        blockDim.y * (head_dim / BRTCH + BSMP) +
                                    (local_batch_idx + block_batch_idx) *
                                        (head_dim / BRTCH + BSMP) +
                                    state_blocklevel_idx + wlch_idx * BWTDH]);
              }
            }
            __syncthreads();
          }
        }
      }
      __syncthreads();
      // store block-level d_state results to global memory for accumulation
      if (BRTCG > 1) {
        const uint state_overcount_warp_idx = (threadIdx.x % warpSize) / BWTDH;
        const uint state_overcount_warp_dim = warpSize / BWTDH;
        const uint state_overcount_block_idx =
            (threadIdx.x / (head_dim * warpSize / BRTCH / BWTDH / BWLCH));
        const uint state_overcount_block_dim =
            (blockDim.x / (head_dim * warpSize / BRTCH / BWTDH / BWLCH));

#pragma unroll
        for (uint local_it = 0;
             local_it <
             CEIL_DIV(BWLCB * BWTDB * BWLCH * BWTDH, BWTCG * WARP_SIZE);
             local_it++) {
          const uint local_total_it =
              local_it * state_overcount_warp_dim * state_overcount_block_dim +
              state_overcount_block_idx * state_overcount_warp_dim +
              state_overcount_warp_idx;

          const uint wlch_idx = local_total_it % BWLCH;
          const uint local_batch_idx = local_total_it / BWLCH;
          const uint local_state_idx = wlch_idx * BWTDH + state_blocklevel_idx;
          const uint global_state_idx =
              multihead_idx + (state_warp_idx - state_warp_local_idx) +
              local_state_idx;
          const uint global_batch_idx = batch_idx + local_batch_idx;
          if (local_batch_idx < BWLCB * BWTDB) {
            d_state_buffer[gate_block_idx * FLASHRNN_HIDDEN_SIZE * batch_dim /
                               BatchIterations +
                           global_batch_idx * FLASHRNN_HIDDEN_SIZE +
                           global_state_idx] =
                mmul_buffer[(local_batch_idx + block_batch_idx) *
                                (head_dim / BRTCH + BSMP) +
                            local_state_idx];
          }
        }

#if FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_GATE *                            \
        FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_HIDDEN >                      \
    1
        gr.sync();
#else
        __syncthreads();
#endif
      }
#pragma unroll
      for (uint local_it = 0; local_it < CEIL_DIV(BWLCB * BWTDB * BWLCH * BWTDH,
                                                  BRTCG * BWTCG * WARP_SIZE);
           local_it++) {
        const uint local_total_it = local_it * gate_grid_dim * overcount +
                                    gate_block_idx * overcount + overcount_idx;

        const uint wlch_idx = local_total_it % BWLCH;
        const uint local_batch_idx = local_total_it / BWLCH;
        const uint local_state_idx = threadIdx.x % (head_dim / BRTCH / BWLCH) +
                                     wlch_idx * (head_dim / BRTCH / BWLCH);
        const uint global_state_idx = multihead_idx +
                                      (state_warp_idx - state_warp_local_idx) +
                                      local_state_idx;

        if (local_batch_idx < BWLCB * BWTDB) {
          float acc = mmul_buffer[(local_batch_idx + block_batch_idx) *
                                      (head_dim / BRTCH + BSMP) +
                                  local_state_idx];
#pragma unroll
          for (uint acc_idx = 1; acc_idx < BRTCG; acc_idx++) {
            const uint int_acc_idx = (acc_idx + gate_block_idx) % BRTCG;
            acc = add_g(
                acc,
                type2float(d_state_buffer[int_acc_idx * FLASHRNN_HIDDEN_SIZE *
                                              batch_dim / BatchIterations +
                                          (batch_idx + local_batch_idx) *
                                              FLASHRNN_HIDDEN_SIZE +
                                          global_state_idx]));
          }
          ds_inout_local[local_it][0] = float2type<FLASHRNN_DTYPE_S>(
              add_g(type2float(ds_inout_local[local_it][0]), acc));
        }
      }

      rgate_offset -= FLASHRNN_NUM_GATES_R * FLASHRNN_HIDDEN_SIZE * batch_dim;
      igate_offset -= FLASHRNN_NUM_GATES_I * FLASHRNN_HIDDEN_SIZE * batch_dim;
      bgate_offset -= (FLASHRNN_NUM_GATES_T - FLASHRNN_NUM_GATES_I) *
                      FLASHRNN_HIDDEN_SIZE * batch_dim;
      state_offset -= FLASHRNN_HIDDEN_SIZE * batch_dim;

    } // end of time loop
#pragma unroll
    for (uint local_it = 0; local_it < CEIL_DIV(BWLCB * BWTDB * BWLCH * BWTDH,
                                                BRTCG * BWTCG * WARP_SIZE);
         local_it++) {
      const uint local_total_it = local_it * gate_grid_dim * overcount +
                                  gate_block_idx * overcount + overcount_idx;

      const uint wlch_idx = local_total_it % BWLCH;
      const uint local_batch_idx = local_total_it / BWLCH;
      const uint local_state_idx = threadIdx.x % (head_dim / BRTCH / BWLCH) +
                                   wlch_idx * (head_dim / BRTCH / BWLCH);
      const uint global_state_idx = multihead_idx +
                                    (state_warp_idx - state_warp_local_idx) +
                                    local_state_idx;

      const uint global_batch_idx =
          batch_idx + batch_it * blockDim.y * gridDim.y * BWLCB * BWTDB +
          local_batch_idx;
      if (local_batch_idx < BWLCB * BWTDB) {
        for (uint sub_state_idx = 0; sub_state_idx < FLASHRNN_NUM_STATES;
             sub_state_idx++) {
          d_states[FLASHRNN_HIDDEN_SIZE * global_batch_idx + global_state_idx +
                   sub_state_idx * FLASHRNN_HIDDEN_SIZE * batch_dim] =
              ds_inout_local[local_it][sub_state_idx];
        }
      }
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

struct BackwardPass::private_data {
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

  cudaStream_t *stream_K;
  cudaEvent_t *event_K;
};

BackwardPass::BackwardPass(const int batch_size, const int hidden_size,
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

  uint num_multihead_streams =
      num_heads / FLASHRNN_BACKWARD_MULTIHEAD_TILING_COUNT;
  data_->event_K =
      (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * num_multihead_streams);
  data_->stream_K =
      (cudaStream_t *)malloc(sizeof(cudaStream_t) * num_multihead_streams);

  for (int i = 0; i < num_multihead_streams; i++) {
    cudaStreamCreate(&data_->stream_K[i]);
    cudaEventCreateWithFlags(&data_->event_K[i], cudaEventDisableTiming);
  }
}

BackwardPass::~BackwardPass() {
  for (int i = _NUM_BLAS_STREAMS - 1; i >= 0; i--) {
    cudaStreamSynchronize(data_->stream_b[i]);
    cublasDestroy(data_->blas_handle_b[i]);
    cudaEventDestroy(data_->event_b[i]);
    cudaStreamDestroy(data_->stream_b[i]);
  }
  uint num_multihead_streams =
      data_->num_heads / FLASHRNN_BACKWARD_MULTIHEAD_TILING_COUNT;
  for (int i = 0; i < num_multihead_streams; i++) {
    cudaStreamSynchronize(data_->stream_K[i]);
    cudaEventDestroy(data_->event_K[i]);
    cudaStreamDestroy(data_->stream_K[i]);
  }
  free(data_->stream_K);
  free(data_->event_K);
  delete data_;
}

void BackwardPass::Set(const int batch_size, const int hidden_size,
                       const int num_heads, const cublasHandle_t &blas_handle,
                       const cudaStream_t &stream) {
  data_->batch_size = batch_size;
  data_->hidden_size = hidden_size;
  data_->main_blas_handle = blas_handle;
  data_->main_stream = stream;

  // reallocate for new number of heads
  if (num_heads != data_->num_heads) {
    uint num_multihead_streams =
        data_->num_heads / FLASHRNN_BACKWARD_MULTIHEAD_TILING_COUNT;
    for (int i = 0; i < num_multihead_streams; i++) {
      cudaStreamSynchronize(data_->stream_K[i]);
      cudaEventDestroy(data_->event_K[i]);
      cudaStreamDestroy(data_->stream_K[i]);
    }
    num_multihead_streams =
        num_heads / FLASHRNN_BACKWARD_MULTIHEAD_TILING_COUNT;
    free(data_->stream_K);
    free(data_->event_K);
    data_->num_heads = num_heads;
    data_->event_K =
        (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * num_multihead_streams);
    data_->stream_K =
        (cudaStream_t *)malloc(sizeof(cudaStream_t) * num_multihead_streams);
    for (int i = 0; i < num_multihead_streams; i++) {
      cudaStreamCreate(&data_->stream_K[i]);
      cudaEventCreateWithFlags(&data_->event_K[i], cudaEventDisableTiming);
    }
  }
}

int BackwardPass::Run(const int steps, const FLASHRNN_DTYPE_R *R_t,
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

  const cudaStream_t *stream_K = data_->stream_K;
  const cudaEvent_t *event_K = data_->event_K;

  // const int NH = batch_size * hidden_size;
  cudaStream_t save_blas_stream;
  bool use_blas_input_stream = false;
  if (cublasGetStream(blas_handle, &save_blas_stream) ==
      CUBLAS_STATUS_SUCCESS) {
    use_blas_input_stream = true;
  } else {
    use_blas_input_stream = false;
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

  const dim3 blockDim(WARP_SIZE * FLASHRNN_BACKWARD_WARP_TILING_COUNT_GATE *
                          head_dim / recurrent_tiling_count_hidden /
                          FLASHRNN_BACKWARD_WARP_LOOPING_COUNT_HIDDEN /
                          FLASHRNN_BACKWARD_WARP_TILING_DIM_HIDDEN,
                      FLASHRNN_BACKWARD_WARP_TILING_COUNT_BATCH, 1);

  const dim3 gridDim(recurrent_tiling_count_hidden,
                     FLASHRNN_BACKWARD_BLOCK_TILING_COUNT_BATCH,
                     FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_GATE *
                         FLASHRNN_BACKWARD_MULTIHEAD_TILING_COUNT);

  int gate_dim_per_block = (FLASHRNN_NUM_GATES_R * head_dim /
                                FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_GATE -
                            (FLASHRNN_BACKWARD_WARP_TILING_COUNT_GATE *
                             FLASHRNN_BACKWARD_WARP_TILING_DIM_GATE *
                             FLASHRNN_BACKWARD_WARP_RECURRENT_CACHED_GATE));

  const uint sharedMemorySizeR =
      (gate_dim_per_block > 0)
          ? sizeof(DTYPE) *
                (head_dim / FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_HIDDEN *
                 (gate_dim_per_block + FLASHRNN_BACKWARD_SHARED_MEMORY_PADDING))
          : 0;

  const uint sharedMemorySizeMatmul =
      sizeof(ACC_DTYPE) *
      (FLASHRNN_BACKWARD_WARP_TILING_DIM_BATCH *
       FLASHRNN_BACKWARD_WARP_TILING_COUNT_BATCH *
       FLASHRNN_BACKWARD_WARP_TILING_COUNT_GATE *
       FLASHRNN_BACKWARD_WARP_LOOPING_COUNT_BATCH *
       (head_dim / FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_HIDDEN +
        FLASHRNN_BACKWARD_SHARED_MEMORY_PADDING));
  const uint sharedMemorySize = sharedMemorySizeR + sharedMemorySizeMatmul;

#ifdef DEBUG
  fprintf(stderr, "Pre-Calc Shared Memory Size: %d from %d, %d, %ld\n",
          sharedMemorySize,
          hidden_size / FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_HIDDEN *
              (FLASHRNN_NUM_GATES_R * hidden_size /
                   FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_GATE +
               FLASHRNN_BACKWARD_SHARED_MEMORY_PADDING),
          FLASHRNN_BACKWARD_WARP_TILING_DIM_BATCH *
              FLASHRNN_BACKWARD_WARP_LOOPING_COUNT_HIDDEN *
              (hidden_size / FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_HIDDEN +
               FLASHRNN_BACKWARD_SHARED_MEMORY_PADDING),
          sizeof(ACC_DTYPE) * FLASHRNN_BACKWARD_WARP_TILING_DIM_BATCH *
              FLASHRNN_BACKWARD_WARP_TILING_COUNT_BATCH *
              FLASHRNN_BACKWARD_WARP_TILING_COUNT_GATE *
              FLASHRNN_BACKWARD_WARP_LOOPING_COUNT_BATCH *
              (head_dim / FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_HIDDEN +
               FLASHRNN_BACKWARD_SHARED_MEMORY_PADDING));
  fprintf(stderr, "Par: %d, %d, %d, %d, %d\n", hidden_size,
          FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_HIDDEN, FLASHRNN_NUM_GATES_R,
          FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_GATE,
          FLASHRNN_BACKWARD_SHARED_MEMORY_PADDING);
#endif
  int maxActiveBlocks;

  // define kernel and increase shared memory from default
  auto kernel = FLASHRNNCellFusedBackward;
  cudaError_t err = cudaSuccess;

  err = cudaFuncSetAttribute(kernel,
                             cudaFuncAttributePreferredSharedMemoryCarveout,
                             cudaSharedmemCarveoutMaxShared);
  if (err != cudaSuccess) {
    fprintf(stderr, "Error setting shared mem attribute carveout\n");
  }
  err = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemorySize);
  if (err != cudaSuccess) {
    fprintf(stderr, "Error setting shared mem attribute size\n");
  }

#ifdef DEBUG
  fprintf(stderr,
          "Values: RTCG: %d, RTCH: %d, WTCG: %d, WTCB: "
          "%d, WTCH: %d\n",
          FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_GATE,
          FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_HIDDEN,
          FLASHRNN_BACKWARD_WARP_TILING_COUNT_GATE,
          FLASHRNN_BACKWARD_WARP_TILING_COUNT_BATCH,
          FLASHRNN_BACKWARD_WARP_LOOPING_COUNT_HIDDEN);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor2(blockDim, sharedMemorySize,
                                                 (void *)kernel);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxActiveBlocks, (void *)kernel, blockDim.x, sharedMemorySize);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  fprintf(stderr,
          "Multiprocessors: %d, Max active Blocks: %d, "
          "Shared Mem per "
          "Block "
          "%lu, per MP: %lu\n",
          prop.multiProcessorCount, maxActiveBlocks, prop.sharedMemPerBlock,
          prop.sharedMemPerMultiprocessor);
  fprintf(stderr, "gridDim: %d, %d, %d, blockDim: %d, %d, %d\n", gridDim.x,
          gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
  fprintf(stderr, "R_block_tile size: %d, %d, %d\n",
          hidden_size / FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_HIDDEN,
          FLASHRNN_NUM_GATES_R * hidden_size /
              FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_GATE,
          sharedMemorySizeR);
  fprintf(stderr, "MMUL BUF SIZE (in bfloat16s): %d, %d\n",
          2 * FLASHRNN_BACKWARD_WARP_TILING_DIM_BATCH *
              FLASHRNN_BACKWARD_WARP_LOOPING_COUNT_HIDDEN *
              (FLASHRNN_HIDDEN_SIZE / FLASHRNN_NUM_HEADS /
                   FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_HIDDEN +
               FLASHRNN_BACKWARD_SHARED_MEMORY_PADDING),
          sharedMemorySizeMatmul);
  fprintf(stderr, "Pre-Kernel launch with shared mem: %d\n", sharedMemorySize);
  fprintf(stderr, "STEPS: %d\n", steps);

#endif

  cudaEventRecord(event_b[0], stream);
  cudaEventRecord(event_b[1], save_blas_stream);
  for (uint i = 0; i < num_heads / FLASHRNN_BACKWARD_MULTIHEAD_TILING_COUNT;
       i++) {
    const uint head_idx =
        i * head_dim * FLASHRNN_BACKWARD_MULTIHEAD_TILING_COUNT;
    cudaStreamWaitEvent(stream_K[i], event_b[0], 0);
    if (use_blas_input_stream) {
      cudaStreamWaitEvent(stream_K[i], event_b[1], 0);
    }
#if FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_GATE *                            \
        FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_HIDDEN >                      \
    1
    const void *R_t_h = (R_t + FLASHRNN_NUM_GATES_R * head_idx * head_dim);
    const void *b_h = (b + FLASHRNN_NUM_GATES_T * head_idx);
    const void *s_h = (s + head_idx);
    const void *g_r_h = (g_r + FLASHRNN_NUM_GATES_R * head_idx);
    const void *g_i_h = (g_i + FLASHRNN_NUM_GATES_I * head_idx);
    const void *g_b_h =
        (g_bias + (FLASHRNN_NUM_GATES_T - FLASHRNN_NUM_GATES_I) * head_idx);
    const void *ds_new_h = (ds_new + head_idx);
    const void *ds_h = (ds + head_idx);
    const void *d_state_buffer_h = (d_state_buffer + head_idx);

    void *kernelArgs[] = {(void *)&steps,
                          (void *)&batch_size,
                          (void *)&R_t_h,
                          (void *)&b_h,
                          (void *)&s_h,
                          (void *)&g_r_h,
                          (void *)&g_i_h,
                          (void *)&g_b_h,
                          (void *)&ds_new_h,
                          (void *)&ds_h,
                          (void *)&d_state_buffer_h};
    err =
        cudaLaunchCooperativeKernel((void *)kernel, gridDim, blockDim,
                                    kernelArgs, sharedMemorySize, stream_K[i]);
#else
    kernel<<<gridDim, blockDim, sharedMemorySize, stream_K[i]>>>(
        steps, batch_size, R_t + FLASHRNN_NUM_GATES_R * head_idx * head_dim,
        b + FLASHRNN_NUM_GATES_T * head_idx, s + head_idx,
        g_r + FLASHRNN_NUM_GATES_R * head_idx,
        g_i + FLASHRNN_NUM_GATES_I * head_idx,
        g_bias + (FLASHRNN_NUM_GATES_T - FLASHRNN_NUM_GATES_I) * head_idx,
        ds_new + head_idx, ds + head_idx, d_state_buffer + head_idx);

#endif
    cudaEventRecord(event_K[i], stream_K[i]);

    for (uint j = 0; j < _NUM_BLAS_STREAMS; j++) {
      cudaStreamWaitEvent(stream_b[j], event_K[i], 0);
    }
  }
  cudaDeviceSynchronize();

  err = cudaPeekAtLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Error after backward kernel launch: %s\n",
            cudaGetErrorString(err));
    fprintf(stderr,
            "Values: RTCG: %d, RTCH: %d, WTCG: % d, BCB: "
            "% d, WTCH: % d\n ",
            FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_GATE,
            FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_HIDDEN,
            FLASHRNN_BACKWARD_WARP_TILING_COUNT_GATE,
            FLASHRNN_BACKWARD_BLOCK_TILING_COUNT_BATCH,
            FLASHRNN_BACKWARD_WARP_LOOPING_COUNT_HIDDEN);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor2(blockDim, sharedMemorySize,
                                                   (void *)kernel);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks, (void *)kernel, blockDim.x, sharedMemorySize);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    fprintf(stderr,
            "Multiprocessors: %d, Max active Blocks: %d, "
            "Shared Mem per "
            "Block "
            "%lu, per MP: %lu\n",
            prop.multiProcessorCount, maxActiveBlocks, prop.sharedMemPerBlock,
            prop.sharedMemPerMultiprocessor);
    fprintf(stderr, "gridDim: %d, %d, %d, blockDim: %d, %d, %d\n", gridDim.x,
            gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
    fprintf(stderr, "R_block_tile size: %d, %d (%d)\n",
            FLASHRNN_NUM_GATES_R * head_dim /
                FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_GATE,
            hidden_size / FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_HIDDEN,
            sharedMemorySizeR);
    fprintf(stderr, "MMUL BUF SIZE: %d (%lu * %d * %d * %d * %d * (%d + %d))\n",
            sharedMemorySizeMatmul, sizeof(ACC_DTYPE),
            FLASHRNN_BACKWARD_WARP_TILING_DIM_BATCH,
            FLASHRNN_BACKWARD_WARP_TILING_COUNT_BATCH,
            FLASHRNN_BACKWARD_WARP_TILING_COUNT_GATE,
            FLASHRNN_BACKWARD_WARP_LOOPING_COUNT_BATCH,
            head_dim / FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_HIDDEN,
            FLASHRNN_BACKWARD_SHARED_MEMORY_PADDING);
    fprintf(stderr, "Backward Pre-Kernel launch with shared mem: %d\n",
            sharedMemorySize);

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
  err = cudaPeekAtLastError();
  if (err != cudaSuccess) {
    return 1;
  }

  for (uint j = 0; j < _NUM_BLAS_STREAMS; j++) {
    cudaEventRecord(event_b[j], stream_b[j]);
    if (use_blas_input_stream) {
      cudaStreamWaitEvent(save_blas_stream, event_b[j]);
    }
    cudaStreamWaitEvent(stream, event_b[j]);
  }
  if (use_blas_input_stream) {
    cublasSetStream(blas_handle, save_blas_stream);
  }
  return 0;
}

} // namespace flashrnn_fused
