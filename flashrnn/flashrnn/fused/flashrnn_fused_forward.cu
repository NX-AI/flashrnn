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

// fused kernel using 16x16x16 MM

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>

#include "../util/blas.h"
#include "../util/cuda_error.h"
#include "../util/inline_ops.cuh"
#include "flashrnn.h"
#include <cooperative_groups.h>
#include <driver_types.h>
#include <mma.h>
#include <stdio.h>

#ifndef _FLASHRNN_POINTWISE_INCLUDED
#include "flashrnn_fused_pointwise_base.cuh"
#endif

#define CEIL_DIV(a, b) (((a) + (b)-1) / (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

// #define DEBUG

namespace {

namespace cg = cooperative_groups;
using namespace nvcuda;

// gate order: i f z o
// FLASHRNN_NUM_GATES_R:     1
//             1 - - -
// FLASHRNN_NUM_GATES_I:     3
//             - 1 1 1

// dimensions
// G: # gates
// FLASHRNN_NUM_GATES_R: # recurrent gates per hidden dimensions (1 for lstmhin,
// 4 for slstm) FLASHRNN_NUM_GATES_I: # gates from input FLASHRNN_NUM_GATES_T: #
// total gates S: # states T: # time steps B: # batch dim H: # hidden dim I: #
// input dim

// General naming convention: dim = real size in memory, count = number along
// axis -> high level dim = count * dim
// -> tile dim = total dim / tile count

#ifndef FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE

// optimized for hidden size 1024
#define FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN 1 // FRTCH 16?
#define FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE 64  // FRTCG 1024 best 64
#define FLASHRNN_FORWARD_BLOCK_TILING_COUNT_BATCH 1      // Btcb
// means extra warps for threads
#define FLASHRNN_FORWARD_WARP_TILING_COUNT_BATCH 1 // Wtcb
// means each warp loops over batches stored in additional shared memory
#define FLASHRNN_FORWARD_WARP_LOOPING_COUNT_BATCH 1      // Wlcp
#define FLASHRNN_FORWARD_WARP_LOOPING_COUNT_GATE 1       // FWLCG
#define FLASHRNN_FORWARD_WARP_TILING_COUNT_HIDDEN 4      // FWTCH 1024 best 8
#define FLASHRNN_FORWARD_WARP_RECURRENT_CACHED_HIDDEN 16 // FWRCH 1024 best 8

#define FLASHRNN_FORWARD_MULTIHEAD_TILING_COUNT 1
#define FLASHRNN_FORWARD_SHARED_MEMORY_PADDING 8

#define FLASHRNN_HIDDEN_SIZE 1024
#define FLASHRNN_NUM_HEADS 1

#define FLASHRNN_FORWARD_WARP_TILING_DIM_BATCH 8 // FWTDB
#define FLASHRNN_FORWARD_WARP_TILING_DIM_GATE 32 // FWTDG

#endif

#define FRTCH FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN
#define FRTCG FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE
#define FBTCB FLASHRNN_FORWARD_BLOCK_TILING_COUNT_BATCH
#define FWTCB FLASHRNN_FORWARD_WARP_TILING_COUNT_BATCH
#define FWLCB FLASHRNN_FORWARD_WARP_LOOPING_COUNT_BATCH
#define FWLCG FLASHRNN_FORWARD_WARP_LOOPING_COUNT_GATE
#define FWTCH FLASHRNN_FORWARD_WARP_TILING_COUNT_HIDDEN
#define FWRCH FLASHRNN_FORWARD_WARP_RECURRENT_CACHED_HIDDEN
#define FMTC FLASHRNN_FORWARD_MULTIHEAD_TILING_COUNT
#define FSMP FLASHRNN_FORWARD_SHARED_MEMORY_PADDING
#define FWTDB FLASHRNN_FORWARD_WARP_TILING_DIM_BATCH
#define FWTDG FLASHRNN_FORWARD_WARP_TILING_DIM_GATE
#define FWTDH FLASHRNN_FORWARD_WARP_TILING_DIM_HIDDEN

#ifdef FLASHRNN_USE_DTYPE_FLOAT32
#define MAT_DTYPE wmma::precision::tf32
#define DTYPE float
#define ACC_DTYPE float
#endif
#ifdef FLASHRNN_USE_DTYPE_FLOAT16
#define MAT_DTYPE __half
#define DTYPE __half
#define ACC_DTYPE __half
#endif
#ifdef FLASHRNN_USE_DTYPE_BFLOAT16
#define MAT_DTYPE __nv_bfloat16
#define DTYPE __nv_bfloat16
#define ACC_DTYPE float
#endif

#define HS FLASHRNN_HIDDEN_SIZE
#define NH FLASHRNN_NUM_HEADS

#define WARP_SIZE 32

// #endif
#define _FLOAT4FACTOR 8

#define _FUSED_KERNEL_MAX_THREADS                                              \
  (WARP_SIZE * FLASHRNN_FORWARD_WARP_TILING_COUNT_HIDDEN *                     \
   FLASHRNN_NUM_GATES_R * FLASHRNN_HIDDEN_SIZE / FLASHRNN_NUM_HEADS /          \
   FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE /                              \
   FLASHRNN_FORWARD_WARP_LOOPING_COUNT_GATE *                                  \
   FLASHRNN_FORWARD_WARP_TILING_COUNT_BATCH /                                  \
   FLASHRNN_FORWARD_WARP_TILING_DIM_GATE)

#define _FUSED_KERNEL_MIN_BLOCKS                                               \
  (FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE *                              \
   FLASHRNN_FORWARD_BLOCK_TILING_COUNT_BATCH *                                 \
   FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN *                            \
   FLASHRNN_FORWARD_MULTIHEAD_TILING_COUNT)

template <bool Training>
__global__ void __launch_bounds__(_FUSED_KERNEL_MAX_THREADS,
                                  _FUSED_KERNEL_MIN_BLOCKS)
    FLASHRNNCellFusedForward(
        const uint steps, const uint batch_dim,
        const FLASHRNN_DTYPE_W *Wx, // Precomputed (Wx) vector [T, B, igate * H]
        const FLASHRNN_DTYPE_R *R,  // recurrect matrix head_dim x head_dim [H,
                                    // FLASHRNN_NUM_GATES_R * H]
        const FLASHRNN_DTYPE_B
            *b, // Bias for gates [G, FLASHRNN_NUM_GATES_T * H]
        FLASHRNN_DTYPE_S *states, // states [S, T + 1, B, H]
        FLASHRNN_DTYPE_G
            *g_r_out, // Output activations (Wx + Ry + b) [], also
                      // contains gate values [T, G-1, B, H] other gates
        FLASHRNN_DTYPE_G
            *g_i_out, // [FLASHRNN_NUM_GATES_T, T, B, H]?  input gate
        ACC_DTYPE *gate_buffer) {

  const uint head_dim = FLASHRNN_HIDDEN_SIZE / FLASHRNN_NUM_HEADS;
  // assuming at least 8 as a batch size, at least 32 as a hidden dim
  // this is necessary for mm
  // each thread takes a tile of (8 x 32) of the pre-activations of one gate,
  // i.e. a (8 x 32) tile of the outputs
  const uint hidden_grid_dim = FRTCH; // equals to FRTCH
#ifdef DEBUG
  if ((threadIdx.x == 0) && (hidden_grid_dim != gridDim.z / FMHTC)) {
    printf("ERROR for hidden_grid_dim: %d, %d, %d\n", gridDim.z, FMHTC, FRTCH);
  }
#endif

  const uint hidden_block_idx = blockIdx.z % hidden_grid_dim;
  const uint multihead_idx = blockIdx.z / hidden_grid_dim * head_dim;

  /// tile of R within head_dim / Rtdh, FLASHRNN_NUM_GATES_R * head_dim / Rtdg
  extern __shared__ float4 sbuf[];
  FLASHRNN_DTYPE_R *R_shared = (FLASHRNN_DTYPE_R *)sbuf;

  FLASHRNN_DTYPE_S
  states_local[CEIL_DIV(FLASHRNN_FORWARD_WARP_LOOPING_COUNT_BATCH *
                            FLASHRNN_FORWARD_WARP_TILING_DIM_BATCH *
                            FLASHRNN_FORWARD_WARP_TILING_DIM_GATE *
                            FLASHRNN_FORWARD_WARP_LOOPING_COUNT_GATE,
                        FLASHRNN_NUM_GATES_R * FRTCH * FWTCH * WARP_SIZE)]
              [FLASHRNN_NUM_STATES];
  FLASHRNN_DTYPE_B biases_local[FLASHRNN_FORWARD_WARP_LOOPING_COUNT_BATCH]
                               [FLASHRNN_NUM_GATES_T];

  // matrix multiplication buffer of size (batch_dim x head_dim / Rtdg)
  int head_dim_per_block_shared =
      ((int)(head_dim / FRTCH) - (int)(FWTDH * FWTCH * FWRCH));
  if (head_dim_per_block_shared < 0) {
    head_dim_per_block_shared = 0;
  }
  ACC_DTYPE *mmul_buffer =
      (ACC_DTYPE *)(((FLASHRNN_DTYPE_R *)(sbuf)) +
                    (head_dim_per_block_shared + FSMP) *
                        (FLASHRNN_NUM_GATES_R * head_dim / FRTCG));
  if (head_dim_per_block_shared == 0) {
    mmul_buffer = (ACC_DTYPE *)sbuf;
  }

  const uint BatchIterations =
      batch_dim / FWTDB / FWLCB / gridDim.y / blockDim.y;
  const uint batch_idx =
      FWLCB * FWTDB * (blockIdx.y * blockDim.y + threadIdx.y);
  const uint block_batch_idx = FWLCB * FWTDB * threadIdx.y;
  const uint gate_warp_idx = FWTDG * FWLCG *
                             ((blockDim.x / FWTCH * blockIdx.x +
                               (threadIdx.x % (blockDim.x / FWTCH))) /
                              warpSize);

  const uint gate_warp_local_idx =
      FWTDG * FWLCG * ((threadIdx.x % (blockDim.x / FWTCH)) / warpSize);
  const uint gate_blocklevel_idx = gate_warp_local_idx + threadIdx.x % FWTDG;
  const uint gate_warp_overcount = warpSize / FWTDG;

  const uint rgate_dim = FLASHRNN_NUM_GATES_R * head_dim;

  if (gate_warp_idx < rgate_dim && batch_idx < batch_dim) {
    uint wtch_idx = threadIdx.x / (blockDim.x / FWTCH);

    const uint B_H = batch_dim * FLASHRNN_HIDDEN_SIZE;
    FLASHRNN_DTYPE_A gates[FLASHRNN_NUM_GATES_T];
#if FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE *                             \
        FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN >                       \
    1
    cg::grid_group gr = cg::this_grid();
#endif

#if (FLASHRNN_FORWARD_WARP_RECURRENT_CACHED_HIDDEN > 0)
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, FWTDB, FWTDG, FWTDH,
                           MAT_DTYPE, nvcuda::wmma::col_major>
        b_frag_cache[FWLCG][FWRCH];

    // store R to registers
    if (FWRCH > 0) {
#pragma unroll
      for (uint wlcg_idx = 0; wlcg_idx < FWLCG; wlcg_idx++) {
        for (uint wrch_idx = 0; wrch_idx < FWRCH; wrch_idx++) {
          uint R_offset =
              FLASHRNN_NUM_GATES_R * multihead_idx * head_dim +
              (blockIdx.x * FLASHRNN_NUM_GATES_R * head_dim / FRTCG +
               gate_warp_local_idx + wlcg_idx * FWTDG) *
                  head_dim +
              hidden_block_idx * (head_dim / FRTCH) +
              wtch_idx * (head_dim / FRTCH / FWTCH) + wrch_idx * FWTDH;
          nvcuda::wmma::load_matrix_sync(b_frag_cache[wlcg_idx][wrch_idx],
                                         R + R_offset, head_dim);
        }
      }
    }
#endif
    // move R to shared mem
    uint local_hidden_offset = FWRCH * FWTDH;
    uint local_hidden_warp_dim = head_dim / FRTCH / FWTCH;
    if (head_dim_per_block_shared > 0) {
#ifdef DEBUG
      if ((threadIdx.x == 0) && (blockIdx.x == 0)) {
        printf("Assigning R, head_dim_block_shared: %d, local_hidden_offset: "
               "%d, local_hidden_warp_dim: %d\n",
               head_dim_per_block_shared, local_hidden_offset,
               local_hidden_warp_dim);
      }
#endif
#pragma unroll
      for (uint j = 0; j < (head_dim_per_block_shared)*FLASHRNN_NUM_GATES_R *
                               head_dim / FRTCG;
           j += blockDim.y * blockDim.x) {
        uint local_linear_idx = (j + threadIdx.y * blockDim.x + threadIdx.x);
        uint local_gate_idx = local_linear_idx / (head_dim_per_block_shared);
        uint local_hidden_idx =
            (local_linear_idx) % (head_dim_per_block_shared);
        uint local_hidden_warp_idx =
            local_hidden_idx % (local_hidden_warp_dim - local_hidden_offset);
        uint local_hidden_block_idx =
            local_hidden_idx / (local_hidden_warp_dim - local_hidden_offset);
        uint global_idx =
            FLASHRNN_NUM_GATES_R * multihead_idx * head_dim +
            (blockIdx.x * FLASHRNN_NUM_GATES_R * head_dim / FRTCG +
             local_gate_idx) *
                head_dim + // gate
            hidden_block_idx * (head_dim / FRTCH) +
            local_hidden_offset + local_hidden_warp_idx +
            local_hidden_block_idx * local_hidden_warp_dim; // hidden

        if (local_gate_idx < FLASHRNN_NUM_GATES_R * head_dim / FRTCG) {
          ((R_shared + local_gate_idx * (head_dim_per_block_shared + FSMP) +
            local_hidden_idx))[0] = ((R + global_idx))[0];
        }
      }
    }

    const uint overthreading_idx = threadIdx.x / (head_dim / FRTCG / FWLCG);
    const uint overthreading_count = MAX(
        1, blockDim.x / (head_dim / FRTCG / FWLCG)); // FLASHRNN_NUM_GATES_R *
                                                     // FWTCH * warpSize / FWTDG
    // load biases

    for (uint local_it = 0;
         local_it < CEIL_DIV(FWTDB * FWLCB * FWTDG * FWLCG,
                             FLASHRNN_NUM_GATES_R * FRTCH * FWTCH * WARP_SIZE);
         local_it++) {
      const uint local_total_it =
          local_it * hidden_grid_dim * overthreading_count +
          hidden_block_idx * overthreading_count + overthreading_idx;
      const uint wlcg_idx = local_total_it % FWLCG;
      // convert stored g to DTYPE_G
      const uint local_state_idx = (threadIdx.x % (head_dim / FRTCG / FWLCG) +
                                    wlcg_idx * (head_dim / FRTCG / FWLCG));
      const uint global_state_idx =
          multihead_idx +
          (gate_warp_idx - gate_warp_local_idx) / FLASHRNN_NUM_GATES_R +
          local_state_idx;

      for (uint gate_idx = 0; gate_idx < FLASHRNN_NUM_GATES_T; gate_idx++) {
        biases_local[wlcg_idx][gate_idx] =
            b[FLASHRNN_NUM_GATES_T * global_state_idx + gate_idx];
      }
    }

    __syncthreads();

    // fragments for matrix multiplication
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, FWTDB, FWTDG, FWTDH,
                           MAT_DTYPE, nvcuda::wmma::row_major>
        a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, FWTDB, FWTDG, FWTDH,
                           MAT_DTYPE, nvcuda::wmma::col_major>
        b_frag;

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, FWTDB, FWTDG, FWTDH,
                           ACC_DTYPE>
        c_frag[FWLCB];

    // main loop
    for (uint batch_it = 0; batch_it < BatchIterations; batch_it++) {
      uint input_idx =
          FLASHRNN_NUM_GATES_I *
          (batch_idx + batch_it * blockDim.y * gridDim.y * FWLCB * FWTDB) *
          FLASHRNN_HIDDEN_SIZE; // used for Wx of dimensions [T, B, (G-1), H]
      uint state_offset =
          (batch_idx + batch_it * blockDim.y * gridDim.y * FWLCB * FWTDB) *
          FLASHRNN_HIDDEN_SIZE;

      // load states into local register
      for (uint local_it = 0; local_it < CEIL_DIV(FWLCB * FWTDB * FWTDG * FWLCG,
                                                  FLASHRNN_NUM_GATES_R * FRTCH *
                                                      FWTCH * WARP_SIZE);
           local_it++) {
        const uint local_total_it =
            local_it * hidden_grid_dim * overthreading_count +
            hidden_block_idx * overthreading_count + overthreading_idx;
        const uint wlcg_idx = local_total_it % FWLCG;
        const uint local_batch_idx = local_total_it / FWLCG;

        if (local_batch_idx < FWLCB * FWTDB) {
          // convert stored g to DTYPE_G
          const uint local_state_idx =
              (threadIdx.x % (head_dim / FRTCG / FWLCG) +
               wlcg_idx * (head_dim / FRTCG / FWLCG));
          const uint global_state_idx =
              multihead_idx +
              (gate_warp_idx - gate_warp_local_idx) / FLASHRNN_NUM_GATES_R +
              local_state_idx;

          // load initial states
          const uint state_pos = state_offset +
                                 local_batch_idx * FLASHRNN_HIDDEN_SIZE +
                                 global_state_idx;
          for (uint state_idx = 0; state_idx < FLASHRNN_NUM_STATES;
               state_idx++) {
            states_local[local_it][state_idx] =
                states[state_pos + state_idx * B_H * (steps + 1)];
          }
        }
      }
      // main time loop
      for (uint t = 0; t < steps; t++) {
        // iterate along FWLCG
        for (uint wlcg_idx = 0; wlcg_idx < FWLCG; wlcg_idx++) {

          uint state_offset_loc =
              state_offset + hidden_block_idx * (head_dim / FRTCH);

          if (gate_warp_local_idx + wlcg_idx * FWTDG <
              FLASHRNN_NUM_GATES_R * head_dim / FRTCG) {
// Initialize the output to zero
#pragma unroll
            for (uint local_batch_idx = 0; local_batch_idx < FWLCB;
                 local_batch_idx++) {
              nvcuda::wmma::fill_fragment(c_frag[local_batch_idx], 0.0f);
            }
            // accumulating matrix multiplications
#if FLASHRNN_FORWARD_WARP_RECURRENT_CACHED_HIDDEN > 0
            for (uint wrch_idx = 0; wrch_idx < FWRCH; wrch_idx++) {
              uint midx =
                  wrch_idx * FWTDH + wtch_idx * (head_dim / FRTCH / FWTCH);
              // change loading here
              for (uint local_batch_idx = 0; local_batch_idx < FWLCB;
                   local_batch_idx++) {
                // Load the inputs
                nvcuda::wmma::load_matrix_sync(
                    a_frag,
                    states + local_batch_idx * FWTDB * FLASHRNN_HIDDEN_SIZE +
                        multihead_idx + state_offset_loc + midx,
                    FLASHRNN_HIDDEN_SIZE);

                nvcuda::wmma::mma_sync(c_frag[local_batch_idx], a_frag,
                                       b_frag_cache[wlcg_idx][wrch_idx],
                                       c_frag[local_batch_idx]);
              }
            }
#endif
            uint R_offset = (gate_warp_local_idx + wlcg_idx * FWTDG) *
                            (head_dim_per_block_shared + FSMP);
            for (uint midx =
                     wtch_idx * (head_dim / FRTCH / FWTCH) + FWRCH * FWTDH;
                 midx < (wtch_idx + 1) * (head_dim / FRTCH / FWTCH);
                 midx += FWTDH) {
              uint R_idx =
                  R_offset +
                  (midx -
                   (wtch_idx * (head_dim / FRTCH / FWTCH) + FWRCH * FWTDH)) +
                  wtch_idx * (head_dim / FRTCH / FWTCH - FWRCH * FWTDH);

              nvcuda::wmma::load_matrix_sync(
                  b_frag, R_shared + R_idx, (head_dim_per_block_shared + FSMP));
              // #endif
              for (uint local_batch_idx = 0; local_batch_idx < FWLCB;
                   local_batch_idx++) {
                // Load the inputs
                nvcuda::wmma::load_matrix_sync(
                    a_frag,
                    states + local_batch_idx * FWTDB * FLASHRNN_HIDDEN_SIZE +
                        multihead_idx + state_offset_loc + midx,
                    FLASHRNN_HIDDEN_SIZE);

                nvcuda::wmma::mma_sync(c_frag[local_batch_idx], a_frag, b_frag,
                                       c_frag[local_batch_idx]);
              }
            }

            for (uint local_batch_idx = 0; local_batch_idx < FWLCB;
                 local_batch_idx++) {
              nvcuda::wmma::store_matrix_sync(
                  mmul_buffer +
                      (wtch_idx * (blockDim.y * FWTDB * FWLCB) +
                       block_batch_idx + local_batch_idx * FWTDB) *
                          (FLASHRNN_NUM_GATES_R * head_dim / FRTCG + FSMP) +
                      gate_warp_local_idx + wlcg_idx * FWTDG,
                  c_frag[local_batch_idx],
                  (FLASHRNN_NUM_GATES_R * head_dim / FRTCG + FSMP),
                  nvcuda::wmma::mem_row_major);
            }
          }

          // accumulate in FWTCH dimension
          if (FWTCH > 1) {
            __syncthreads();
          }

          if (gate_warp_local_idx + wlcg_idx * FWTDG <
              FLASHRNN_NUM_GATES_R * head_dim / FRTCG) {
            // accumulate along FWTCH tiling dimension
            for (uint local_batch_idx = wtch_idx * gate_warp_overcount +
                                        (threadIdx.x % warpSize) / FWTDG;
                 local_batch_idx < FWLCB * FWTDB;
                 local_batch_idx += FWTCH * gate_warp_overcount) {
              for (uint local_wtch_idx = 1; local_wtch_idx < FWTCH;
                   local_wtch_idx++) {
                mmul_buffer[(local_batch_idx + block_batch_idx) *
                                (FLASHRNN_NUM_GATES_R * head_dim / FRTCG +
                                 FSMP) +
                            gate_blocklevel_idx + wlcg_idx * FWTDG] =
                    add_g(mmul_buffer[(local_batch_idx + block_batch_idx) *
                                          (FLASHRNN_NUM_GATES_R * head_dim /
                                               FRTCG +
                                           FSMP) +
                                      gate_blocklevel_idx + wlcg_idx * FWTDG],
                          mmul_buffer
                              [(local_batch_idx + block_batch_idx +
                                local_wtch_idx * blockDim.y * FWTDB * FWLCB) *
                                   (FLASHRNN_NUM_GATES_R * head_dim / FRTCG +
                                    FSMP) +
                               gate_blocklevel_idx + wlcg_idx * FWTDG]);
              }
            }
          } else {
#ifdef DEBUG
            printf("Got a problematic matmul index\n");
#endif
          }
        }
        __syncthreads();
#if FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN > 1
        // store in global memory aligned with floats

        const uint gate_overthreading_warp_idx =
            (threadIdx.x % warpSize) / FWTDG;
        const uint gate_overthreading_warp_count = warpSize / FWTDG;
        // const uint gate_overthreading_idx =
        //     threadIdx.x / ((FLASHRNN_NUM_GATES_R * head_dim) / FRTCG);
        const uint gate_overthreading_block_idx =
            (threadIdx.x / ((FLASHRNN_NUM_GATES_R * head_dim * warpSize) /
                            FRTCG / FWTDG / FWLCG));
        // const uint gate_overthreading_block_dim =
        //     (blockDim.x / ((FLASHRNN_NUM_GATES_R * head_dim * warpSize) /
        //                    FRTCG / FWTDG / FWLCG));
        const uint gate_overthreading_count =
            blockDim.x / ((FLASHRNN_NUM_GATES_R * head_dim) / FRTCG / FWLCG);

#pragma unroll
        for (uint local_it = 0;
             local_it <
             CEIL_DIV(FWLCB * FWTDB * FWTDG * FWLCG, FWTCH * WARP_SIZE);
             local_it++) {
          const uint local_total_it =
              local_it * gate_overthreading_count +
              gate_overthreading_block_idx * gate_overthreading_warp_count +
              gate_overthreading_warp_idx;
          const uint wlcg_idx = local_total_it % FWLCG;
          const uint local_batch_idx = local_total_it / FWLCG;
          const uint gate_idx = gate_warp_idx + threadIdx.x % FWTDG;
          if (local_batch_idx < FWLCB * FWTDB) {
            gate_buffer[(hidden_block_idx * FLASHRNN_HIDDEN_SIZE *
                             FLASHRNN_NUM_GATES_R * batch_dim /
                             BatchIterations +
                         (batch_idx + local_batch_idx) *
                             (FLASHRNN_HIDDEN_SIZE * FLASHRNN_NUM_GATES_R) +
                         FLASHRNN_NUM_GATES_R * multihead_idx +
                         wlcg_idx * FWTDG + gate_idx)] =
                ((mmul_buffer +
                  (block_batch_idx + local_batch_idx) *
                      (FLASHRNN_NUM_GATES_R * head_dim / FRTCG + FSMP) +
                  wlcg_idx * FWTDG + gate_blocklevel_idx))[0];
          }
        }
#if FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE *                             \
        FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN >                       \
    1
        gr.sync();
#endif
#endif

// iterate over tile, pointwise operations
#pragma unroll
        for (uint local_it = 0;
             local_it <
             CEIL_DIV(FWLCB * FWTDB * FWTDG * FWLCG,
                      FLASHRNN_NUM_GATES_R * FRTCH * FWTCH * WARP_SIZE);
             local_it++) {
          const uint local_total_it =
              local_it * hidden_grid_dim * overthreading_count +
              hidden_block_idx * overthreading_count + overthreading_idx;
          const uint wlcg_idx = local_total_it % FWLCG;
          const uint local_batch_idx = local_total_it / FWLCG;

          if (local_batch_idx < FWLCB * FWTDB) {
            // convert stored g to DTYPE_G
            const uint local_state_idx =
                (threadIdx.x % (head_dim / FRTCG / FWLCG) +
                 wlcg_idx * (head_dim / FRTCG / FWLCG));
            const uint global_state_idx =
                multihead_idx +
                (gate_warp_idx - gate_warp_local_idx) / FLASHRNN_NUM_GATES_R +
                local_state_idx;

            for (uint gidx = 0; gidx < FLASHRNN_NUM_GATES_T; gidx++) {
              gates[gidx] = float2type<FLASHRNN_DTYPE_A>(
                  type2float(biases_local[wlcg_idx][gidx]));
            }

// // add recurrent contributions
#pragma unroll
            for (uint gidx = 0; gidx < FLASHRNN_NUM_GATES_R; gidx++) {
              float acc = type2float(
                  *(mmul_buffer +
                    (block_batch_idx + local_batch_idx) *
                        (FLASHRNN_NUM_GATES_R * head_dim / FRTCG + FSMP) +
                    gidx + FLASHRNN_NUM_GATES_R * local_state_idx));
#pragma unroll
              for (uint acc_idx = 1; acc_idx < FRTCH; acc_idx++) {
                const uint int_acc_idx = (hidden_block_idx + acc_idx) % FRTCH;
                acc += ((float *)gate_buffer)[(
                    int_acc_idx * FLASHRNN_HIDDEN_SIZE * FLASHRNN_NUM_GATES_R *
                        batch_dim / BatchIterations +
                    (batch_idx + local_batch_idx) *
                        (FLASHRNN_HIDDEN_SIZE * FLASHRNN_NUM_GATES_R) +
                    FLASHRNN_NUM_GATES_R * global_state_idx + gidx)];
              }
              if (!FLASHRNN_SIMPLE_AGG) {
                // save Ry for backward
                g_r_out[B_H * FLASHRNN_NUM_GATES_R * t +
                        (batch_idx +
                         batch_it * blockDim.y * gridDim.y * FWLCB * FWTDB +
                         local_batch_idx) *
                            FLASHRNN_HIDDEN_SIZE * FLASHRNN_NUM_GATES_R +
                        FLASHRNN_NUM_GATES_R * global_state_idx + gidx] =
                    float2type<FLASHRNN_DTYPE_G>(acc);
                acc = FLASHRNNRecurrentActivation(acc, gidx);
              }
              gates[gidx] = float2type<FLASHRNN_DTYPE_A>(
                  add_g(type2float(gates[gidx]), acc));
            }

            for (uint gidx = 0; gidx < FLASHRNN_NUM_GATES_W; gidx++) {
              gates[FLASHRNN_NUM_GATES_I - FLASHRNN_NUM_GATES_W + gidx] =
                  float2type<FLASHRNN_DTYPE_A>(add_g(
                      type2float(gates[FLASHRNN_NUM_GATES_I -
                                       FLASHRNN_NUM_GATES_W + gidx]),
                      type2float(
                          Wx[input_idx +
                             FLASHRNN_NUM_GATES_W * local_batch_idx *
                                 FLASHRNN_HIDDEN_SIZE +
                             FLASHRNN_NUM_GATES_W * global_state_idx + gidx])));
            }

            const uint state_pos = state_offset +
                                   local_batch_idx * FLASHRNN_HIDDEN_SIZE +
                                   global_state_idx;
            // pointwise operations
            FLASHRNNPointwiseForward<Training>(
                states_local[local_it], gates, 1, states + state_pos + B_H,
                states + state_pos + B_H * (steps + 1) + B_H, B_H * (steps + 1),
                g_r_out + B_H * FLASHRNN_NUM_GATES_R * t +
                    (batch_idx +
                     batch_it * blockDim.y * gridDim.y * FWLCB * FWTDB +
                     local_batch_idx) *
                        FLASHRNN_HIDDEN_SIZE * FLASHRNN_NUM_GATES_R +
                    FLASHRNN_NUM_GATES_R * global_state_idx,
                1,
                g_i_out + B_H * FLASHRNN_NUM_GATES_I * t +
                    (batch_idx +
                     batch_it * blockDim.y * gridDim.y * FWLCB * FWTDB +
                     local_batch_idx) *
                        FLASHRNN_HIDDEN_SIZE * FLASHRNN_NUM_GATES_I +
                    FLASHRNN_NUM_GATES_I * global_state_idx,
                1);
          }
        }
#if FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE *                             \
        FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN >                       \
    1
        gr.sync();
#else
        __syncthreads();
#endif

        input_idx += FLASHRNN_NUM_GATES_W * B_H;
        state_offset += B_H;
      }
    }
  }
}

} // anonymous namespace

namespace flashrnn_fused {
struct ForwardPass::private_data {
  bool training;
  int batch_size;
  int hidden_size;
  int num_heads;
  cublasHandle_t main_blas_handle;

  cublasHandle_t *blas_handle_K; // kernels
  cudaStream_t stream;

  cudaStream_t *stream_K;

  cudaEvent_t *event_K;
  cudaEvent_t ready_event;
  cudaEvent_t finished_event;
};

ForwardPass::ForwardPass(const bool training, const int batch_size,
                         const int hidden_size, const int num_heads,
                         const cublasHandle_t &blas_handle,
                         const cudaStream_t &stream)
    : data_(new private_data) {
  data_->training = training;
  data_->batch_size = batch_size;
  data_->hidden_size = hidden_size;
  data_->num_heads = num_heads;
  data_->main_blas_handle = blas_handle;

  uint num_multihead_streams =
      num_heads / FLASHRNN_FORWARD_MULTIHEAD_TILING_COUNT;
  data_->stream_K =
      (cudaStream_t *)malloc(sizeof(cudaStream_t) * num_multihead_streams);
  data_->event_K =
      (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * num_multihead_streams);
  data_->blas_handle_K =
      (cublasHandle_t *)malloc(sizeof(cublasHandle_t) * num_multihead_streams);

  for (uint i = 0; i < num_multihead_streams; i++) {
    cudaStreamCreate(&data_->stream_K[i]);
    cublasCreate(&data_->blas_handle_K[i]);
    cudaEventCreateWithFlags(&data_->event_K[i], cudaEventDisableTiming);
  }

  cudaEventCreateWithFlags(&data_->ready_event, cudaEventDisableTiming);
  cudaEventCreateWithFlags(&data_->finished_event, cudaEventDisableTiming);
}

void ForwardPass::Set(const bool training, const int batch_size,
                      const int hidden_size, const int num_heads,
                      const cublasHandle_t &blas_handle,
                      const cudaStream_t &stream) {
  data_->training = training;
  data_->batch_size = batch_size;
  data_->hidden_size = hidden_size;
  data_->main_blas_handle = blas_handle;
  data_->stream = stream;
  uint num_multihead_streams =
      data_->num_heads / FLASHRNN_FORWARD_MULTIHEAD_TILING_COUNT;
  if (FLASHRNN_NUM_HEADS != data_->num_heads) {
    // deconstruct old allocations
    for (uint i = 0; i < num_multihead_streams; i++) {
      cudaStreamSynchronize(data_->stream_K[i]);
      cudaEventDestroy(data_->event_K[i]);
      cudaStreamDestroy(data_->stream_K[i]);
      cublasDestroy(data_->blas_handle_K[i]);
    }
    num_multihead_streams =
        FLASHRNN_NUM_HEADS / FLASHRNN_FORWARD_MULTIHEAD_TILING_COUNT;
    free(data_->blas_handle_K);
    free(data_->event_K);
    free(data_->stream_K);

    // allocate new allocations
    data_->stream_K =
        (cudaStream_t *)malloc(sizeof(cudaStream_t) * num_multihead_streams);
    data_->event_K =
        (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * num_multihead_streams);
    data_->blas_handle_K = (cublasHandle_t *)malloc(sizeof(cublasHandle_t) *
                                                    num_multihead_streams);

    for (uint i = 0; i < num_multihead_streams; i++) {
      cudaStreamCreate(&data_->stream_K[i]);
      cublasCreate(&data_->blas_handle_K[i]);
      cudaEventCreateWithFlags(&data_->event_K[i], cudaEventDisableTiming);
    }
  }
  data_->num_heads = FLASHRNN_NUM_HEADS;
}

ForwardPass::~ForwardPass() {
  cudaEventDestroy(data_->finished_event);
  cudaEventDestroy(data_->ready_event);

  uint num_multihead_streams =
      data_->num_heads / FLASHRNN_FORWARD_MULTIHEAD_TILING_COUNT;
  for (uint i = 0; i < num_multihead_streams; i++) {
    cudaStreamSynchronize(data_->stream_K[i]);
    cudaEventDestroy(data_->event_K[i]);
    cudaStreamDestroy(data_->stream_K[i]);
    cublasDestroy(data_->blas_handle_K[i]);
  }

  free(data_->blas_handle_K);
  free(data_->event_K);
  free(data_->stream_K);
  delete data_;
}

int ForwardPass::Run(
    const int steps,
    const FLASHRNN_DTYPE_R *R, // Weight matrix for recurrent state (Ry) [y,H*4]
    const FLASHRNN_DTYPE_B *b, // Bias for gates (Wx + Ry + b) [H*4]
    const FLASHRNN_DTYPE_W *x, // Input vector [T,N,C]
    FLASHRNN_DTYPE_S *s,       // Cell states [S+1,N,H]
    FLASHRNN_DTYPE_G *g_r,     // Output vector (Wx + Ry + b) [S,N,H*3])
    FLASHRNN_DTYPE_G *g_i,
    FLASHRNN_ACC_DTYPE *gate_buffer) { // Output vector (Wx + Ry + b) [S,N,H]
  const blas<void>::set_pointer_mode scoped1(data_->main_blas_handle);

  const uint batch_size = data_->batch_size;
  const uint head_dim = data_->hidden_size / FLASHRNN_NUM_HEADS;

  const cublasHandle_t blas_handle = data_->main_blas_handle;

  const cudaStream_t *stream_K = data_->stream_K;
  const cudaEvent_t *event_K = data_->event_K;
  cudaStream_t blas_save_stream;
  uint recurrent_tiling_count_gate = MIN(
      FLASHRNN_NUM_GATES_R * head_dim / FLASHRNN_FORWARD_WARP_TILING_DIM_GATE /
          FLASHRNN_FORWARD_WARP_LOOPING_COUNT_GATE,
      MAX(WARP_SIZE * FLASHRNN_NUM_GATES_R * head_dim / 1024 /
              FLASHRNN_FORWARD_WARP_TILING_DIM_GATE /
              FLASHRNN_FORWARD_WARP_LOOPING_COUNT_GATE,
          FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE)); // because the
                                                          // maximal block
                                                          // size is 1024

  if (recurrent_tiling_count_gate !=
      FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE) {
    fprintf(stderr,
            "The specified forward RECURRENT_TILING_COUNT_GATE should be: %d\n",
            recurrent_tiling_count_gate);
    fprintf(stderr, "Values: RTCG: %d, RTCH: %d, WTCG: %d, BCB: %d, WTCH: %d\n",
            FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE,
            FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN,
            FLASHRNN_FORWARD_WARP_LOOPING_COUNT_GATE,
            FLASHRNN_FORWARD_BLOCK_TILING_COUNT_BATCH,
            FLASHRNN_FORWARD_WARP_TILING_COUNT_HIDDEN);
    return 1;
  }

  const dim3 blockDim(WARP_SIZE * FLASHRNN_FORWARD_WARP_TILING_COUNT_HIDDEN *
                          FLASHRNN_NUM_GATES_R * FLASHRNN_HIDDEN_SIZE /
                          FLASHRNN_NUM_HEADS /
                          FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE /
                          FLASHRNN_FORWARD_WARP_LOOPING_COUNT_GATE /
                          FLASHRNN_FORWARD_WARP_TILING_DIM_GATE,
                      FLASHRNN_FORWARD_WARP_TILING_COUNT_BATCH, 1);

  const dim3 gridDim(recurrent_tiling_count_gate,
                     FLASHRNN_FORWARD_BLOCK_TILING_COUNT_BATCH,
                     FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN *
                         FLASHRNN_FORWARD_MULTIHEAD_TILING_COUNT);

  // Recurrent matrix tile and matmul gate buffer, 2 for float32 of mmul
  // buffer shared memory size is in bytes!, this is why the sizeof is needed

  int head_dim_per_block =
      (head_dim / FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN -
       (FLASHRNN_FORWARD_WARP_TILING_COUNT_HIDDEN *
        FLASHRNN_FORWARD_WARP_TILING_DIM_HIDDEN *
        FLASHRNN_FORWARD_WARP_RECURRENT_CACHED_HIDDEN));
  const uint sharedMemorySizeR =
      (head_dim_per_block > 0)
          ? sizeof(DTYPE) * FLASHRNN_NUM_GATES_R * head_dim /
                recurrent_tiling_count_gate *
                (head_dim_per_block + FLASHRNN_FORWARD_SHARED_MEMORY_PADDING)
          : 0;
  const uint sharedMemorySizeMatmul =
      sizeof(ACC_DTYPE) * FLASHRNN_FORWARD_WARP_TILING_DIM_BATCH *
      FLASHRNN_FORWARD_WARP_LOOPING_COUNT_BATCH *
      FLASHRNN_FORWARD_WARP_TILING_COUNT_BATCH *
      FLASHRNN_FORWARD_WARP_TILING_COUNT_HIDDEN *
      (FLASHRNN_NUM_GATES_R * head_dim / recurrent_tiling_count_gate +
       FLASHRNN_FORWARD_SHARED_MEMORY_PADDING);
  const uint sharedMemorySize = sharedMemorySizeR + sharedMemorySizeMatmul;

#ifdef DEBUG
  printf("Shared Memory Size: %d (= %d (= %d * %d) + "
         "%d))\n",
         sharedMemorySize, sharedMemorySizeR,
         FLASHRNN_NUM_GATES_R * head_dim / recurrent_tiling_count_gate,
         head_dim_per_block, sharedMemorySizeMatmul);
#endif
  int maxActiveBlocks;

  // define kernel and increase shared memory from
  // default
  auto kernel = FLASHRNNCellFusedForward<true>;
  cudaError_t err = cudaSuccess;
  err = cudaFuncSetAttribute(kernel,
                             cudaFuncAttributePreferredSharedMemoryCarveout,
                             cudaSharedmemCarveoutMaxShared);
  if (err != cudaSuccess) {
    fprintf(stderr, "Error setting shared mem attribute carveout");
  }
  err = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemorySize);
  if (err != cudaSuccess) {
    fprintf(stderr, "Error setting shared mem attribute size");
  }

  bool use_blas_input_stream = false;
  if (cublasGetStream(blas_handle, &blas_save_stream) ==
      CUBLAS_STATUS_SUCCESS) {
    use_blas_input_stream = true;
  } else {
    use_blas_input_stream = false;
  }
  cudaEventRecord(event_K[0], data_->stream);
  if (use_blas_input_stream) {
    cudaEventRecord(event_K[0], blas_save_stream);
  }

  for (uint i = 0;
       i < FLASHRNN_NUM_HEADS / FLASHRNN_FORWARD_MULTIHEAD_TILING_COUNT; i++) {
    uint head_idx = i * head_dim * FLASHRNN_FORWARD_MULTIHEAD_TILING_COUNT;
    cudaStreamWaitEvent(stream_K[i], event_K[0]);

#if FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE *                             \
        FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN >                       \
    1
    void *x_h = (void *)(x + FLASHRNN_NUM_GATES_W * head_idx);
    void *g_r_h = (void *)(g_r + FLASHRNN_NUM_GATES_R * head_idx);
    void *g_i_h = (void *)(g_i + FLASHRNN_NUM_GATES_I * head_idx);
    void *R_h = (void *)(R + FLASHRNN_NUM_GATES_R * head_dim * head_idx);
    void *b_h = (void *)(b + FLASHRNN_NUM_GATES_T * head_idx);
    void *s_h = (void *)(s + head_idx);
    void *gate_buffer_h =
        (void *)(gate_buffer + FLASHRNN_NUM_GATES_R * head_idx);

    void *kernelArgs[] = {
        (void *)&steps, (void *)&batch_size, (void *)&x_h,
        (void *)&R_h,   (void *)&b_h,        (void *)&s_h,
        (void *)&g_r_h, (void *)&g_i_h,      (void *)&gate_buffer_h};
    err =
        cudaLaunchCooperativeKernel((void *)kernel, gridDim, blockDim,
                                    kernelArgs, sharedMemorySize, stream_K[i]);
#else

    kernel<<<gridDim, blockDim, sharedMemorySize, stream_K[i]>>>(
        steps, batch_size, x + FLASHRNN_NUM_GATES_W * head_idx,
        R + FLASHRNN_NUM_GATES_R * head_dim * head_idx,
        b + FLASHRNN_NUM_GATES_T * head_idx, s + head_idx,
        g_r + FLASHRNN_NUM_GATES_R * head_idx,
        g_i + FLASHRNN_NUM_GATES_I * head_idx,
        (ACC_DTYPE *)gate_buffer + FLASHRNN_NUM_GATES_R * head_idx);

#endif
    cudaEventRecord(event_K[i], stream_K[i]);
    cudaStreamWaitEvent(data_->stream, event_K[i]);
    if (use_blas_input_stream) {
      cudaStreamWaitEvent(blas_save_stream, event_K[i]);
    }
  }
  if (err == cudaSuccess) {
#ifdef DEBUG
    printf("NO ERROR until after execution\n");
#endif
    err = cudaPeekAtLastError();
  }
  if (err != cudaSuccess) {
    fprintf(stderr, "Error after forward kernel launch: %s\n",
            cudaGetErrorString(err));
    fprintf(stderr,
            "Values: RTCG: %d, RTCH: %d, WTCG: %d, BCB: "
            "%d, WTCH: %d\n",
            FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE,
            FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN,
            FLASHRNN_FORWARD_WARP_LOOPING_COUNT_GATE,
            FLASHRNN_FORWARD_BLOCK_TILING_COUNT_BATCH,
            FLASHRNN_FORWARD_WARP_TILING_COUNT_HIDDEN);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor2(blockDim, sharedMemorySize,
                                                   (void *)kernel);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks, (void *)kernel, blockDim.x, sharedMemorySize);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    fprintf(stderr,
            "Multiprocessors: %d, Max active Blocks: "
            "%d, Shared Mem per Block "
            "%lu, per MP: %lu\n",
            prop.multiProcessorCount, maxActiveBlocks, prop.sharedMemPerBlock,
            prop.sharedMemPerMultiprocessor);
    fprintf(stderr, "gridDim: %d, %d, %d, blockDim: %d, %d, %d\n", gridDim.x,
            gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
    fprintf(stderr, "R_block_tile size: %d, %d -> %d\n",
            FLASHRNN_NUM_GATES_R * head_dim /
                FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE,
            head_dim / FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN,
            sharedMemorySizeR);
    fprintf(stderr, "MMUL BUF SIZE : %d\n", sharedMemorySizeMatmul);
    fprintf(stderr, "Pre-Kernel launch with shared mem: %d\n",
            sharedMemorySize);

    return 1;
  }
  if (use_blas_input_stream) {
    cublasSetStream(blas_handle, blas_save_stream);
  }
  return 0;
}

} // namespace flashrnn_fused
