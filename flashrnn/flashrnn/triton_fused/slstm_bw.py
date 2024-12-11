# Maximilian Beck
import triton.language as tl
import triton
import torch
from triton import OutOfResources
from einops import rearrange
from .triton_utils import torch2triton_dtype, is_power_of_2, next_multiple_of
from typing import Callable

# Dimensions:
# B: batch size
# T: sequence length
# NGI: number of gates that depend on input
# NGR: number of gates that depend on recurrent state
# NH: number of heads
# DH: hidden dimension
# NS: number of states

# Note: NS dimension
# > in NS dimension the first index is the state that is used for recurrent weights


# Note on the kernel:
# we only pass the dimensions and not the stride to the kernel
# inside we compute the strides from the dimensions

# we assume for simplicity: NGR == NGI

# TODO: add autotuning, for now we use a hardcoded configuration, num_stages=1, num_warps=4, siz_B=16
# ENABLE_AUTOTUNING = False

# if ENABLE_AUTOTUNING:
#     configs = [
#         triton.Config({"siz_B": siz_B}, num_stages=s, num_warps=w)
#         for siz_B in [16, 32, 64]
#         for s in [1, 2, 3, 4]
#         for w in [1, 2, 4, 8]
#     ]
# else:
#     configs = [
#         triton.Config({"siz_B": siz_B}, num_stages=s, num_warps=w)
#         for siz_B in [16]
#         for s in [1]
#         for w in [4]
#     ]


@triton.jit
def triton_tanh(x):
    return (1.0 - tl.exp(-2.0 * x)) / (1.0 + tl.exp(-2.0 * x))


# @triton.autotune(configs, key=["siz_B", "T", "B", "NH", "DH"])
@triton.jit
def _backward_sequence_kernel(
    # inputs
    delta_states_all_outside,  # (NH, T, NS=4, B, DH) delta errors from all states
    delta_states_last_outside,  # (NH, NS=4, B, DH) delta errors from the last state
    R,  # (NH, NGR, DHout, DHin) recurrent weights
    states_all,  # (NH, T+1, NS=4, B, DH) all states
    gates_all,  # (NH, T, NGI, B, DH) all gates
    # outputs
    delta_states_initial,  # (NH, NS=4, B, DH) delta errors to initial states
    delta_Wx,  # (NH, T, NGI, B, DH) delta errors to inputs
    delta_R,  # (NH, NGR, DHout, DHin) delta errors to recurrent weights
    delta_b,  # (NH, NGI, DH) delta errors to biases
    # dimensions
    T: tl.constexpr,  # sequence length
    NS: tl.constexpr,  # number of states
    B: tl.constexpr,  # batch size
    NH: tl.constexpr,  # number of heads
    DH: tl.constexpr,  # hidden dimension
    NGI: tl.constexpr,  # number of gates that depend on input
    NGR: tl.constexpr,  # number of gates that depend on recurrent state
    siz_B: tl.constexpr,  # the number of batches per thread block
    # consts
    DTYPE: tl.constexpr = tl.float32,
    backward_recurrent_clip_val: tl.constexpr = -1.0,  # if > 0, clip the recurrent gradients
):
    idx_b_NH, idx_b_B = tl.program_id(0), tl.program_id(1)

    ## compute the strides
    str_matR_B = NH * NGR * DH * DH
    str_matR_NH = NGR * DH * DH
    str_matR_NGR = DH * DH
    str_matStatesAll_NH = (T + 1) * NS * B * DH
    str_matStatesAll_T = NS * B * DH
    str_matGatesAll_NH = T * NGI * B * DH
    str_matGatesAll_T = NGI * B * DH
    str_delta_states_all_outside_NH = T * NS * B * DH
    str_delta_states_all_outside_T = NS * B * DH
    str_matDeltaWx_NH = T * NGI * B * DH
    str_matDeltaWx_T = NGI * B * DH
    ##

    ## load delta errors from last state
    # load matDeltaH from last state
    matDeltaHtrans_last_ptr = tl.make_block_ptr(
        base=delta_states_last_outside + idx_b_NH * NS * B * DH + 0 * B * DH,
        shape=(B, DH),
        strides=(DH, 1),
        offsets=(idx_b_B * siz_B, 0),
        block_shape=(siz_B, DH),
        order=(0, 1),
    )
    matDeltaH_tplus1 = tl.load(matDeltaHtrans_last_ptr).to(tl.float32)  # (siz_B, DH)

    # load matDeltaC from last state
    matDeltaCtrans_last_ptr = tl.make_block_ptr(
        base=delta_states_last_outside + idx_b_NH * NS * B * DH + 1 * B * DH,
        shape=(B, DH),
        strides=(DH, 1),
        offsets=(idx_b_B * siz_B, 0),
        block_shape=(siz_B, DH),
        order=(0, 1),
    )
    matDeltaC_tplus1 = tl.load(matDeltaCtrans_last_ptr).to(tl.float32)  # (siz_B, DH)

    # load matDeltaN_tplus1 from last state
    matDeltaNtrans_last_ptr = tl.make_block_ptr(
        base=delta_states_last_outside + idx_b_NH * NS * B * DH + 2 * B * DH,
        shape=(B, DH),
        strides=(DH, 1),
        offsets=(idx_b_B * siz_B, 0),
        block_shape=(siz_B, DH),
        order=(0, 1),
    )
    matDeltaN_tplus1 = tl.load(matDeltaNtrans_last_ptr).to(tl.float32)  # (siz_B, DH)

    # Note: we do not need to load the m state as the gradients through this state cancel out

    ## load the recurrent weights only once
    matR_i_ptr = tl.make_block_ptr(
        base=R + idx_b_NH * str_matR_NH + 0 * str_matR_NGR,
        shape=(DH, DH),
        strides=(DH, 1),
        offsets=(0, 0),
        block_shape=(DH, DH),
        order=(0, 1),
    )
    matR_i = tl.load(matR_i_ptr)  # (DHout, DHin)

    matR_f_ptr = tl.make_block_ptr(
        base=R + idx_b_NH * str_matR_NH + 1 * str_matR_NGR,
        shape=(DH, DH),
        strides=(DH, 1),
        offsets=(0, 0),
        block_shape=(DH, DH),
        order=(0, 1),
    )
    matR_f = tl.load(matR_f_ptr)  # (DHout, DHin)

    matR_z_ptr = tl.make_block_ptr(
        base=R + idx_b_NH * str_matR_NH + 2 * str_matR_NGR,
        shape=(DH, DH),
        strides=(DH, 1),
        offsets=(0, 0),
        block_shape=(DH, DH),
        order=(0, 1),
    )
    matR_z = tl.load(matR_z_ptr)  # (DHout, DHin)

    matR_o_ptr = tl.make_block_ptr(
        base=R + idx_b_NH * str_matR_NH + 3 * str_matR_NGR,
        shape=(DH, DH),
        strides=(DH, 1),
        offsets=(0, 0),
        block_shape=(DH, DH),
        order=(0, 1),
    )
    matR_o = tl.load(matR_o_ptr)  # (DHout, DHin)

    ## init the delta errors for the recurrent weights and the biases
    # in shared memory
    # we accumulate the gradients in float32
    matDeltaR_i = tl.zeros((DH, DH), dtype=tl.float32)
    matDeltaR_f = tl.zeros((DH, DH), dtype=tl.float32)
    matDeltaR_z = tl.zeros((DH, DH), dtype=tl.float32)
    matDeltaR_o = tl.zeros((DH, DH), dtype=tl.float32)

    vecDeltaB_i = tl.zeros((DH,), dtype=tl.float32)
    vecDeltaB_f = tl.zeros((DH,), dtype=tl.float32)
    vecDeltaB_z = tl.zeros((DH,), dtype=tl.float32)
    vecDeltaB_o = tl.zeros((DH,), dtype=tl.float32)

    ## loop over the sequence from T-1 to 0 (inclduding 0)
    for idx_t in range(T - 1, -1, -1):
        ## load gate activations G for the current time step idx_t
        # gates in float32 as tl.exp and tl.log are not supported in (b)float16
        matG_i_ptr = tl.make_block_ptr(
            base=gates_all
            + idx_b_NH * str_matGatesAll_NH
            + idx_t * str_matGatesAll_T
            + 0 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_b_B * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        matG_ibar = tl.load(matG_i_ptr).to(tl.float32)  # (siz_B, DH)

        matG_f_ptr = tl.make_block_ptr(
            base=gates_all
            + idx_b_NH * str_matGatesAll_NH
            + idx_t * str_matGatesAll_T
            + 1 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_b_B * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        matG_fbar = tl.load(matG_f_ptr).to(tl.float32)  # (siz_B, DH)

        matG_z_ptr = tl.make_block_ptr(
            base=gates_all
            + idx_b_NH * str_matGatesAll_NH
            + idx_t * str_matGatesAll_T
            + 2 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_b_B * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        matG_z = tl.load(matG_z_ptr)  # (siz_B, DH)

        matG_o_ptr = tl.make_block_ptr(
            base=gates_all
            + idx_b_NH * str_matGatesAll_NH
            + idx_t * str_matGatesAll_T
            + 3 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_b_B * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        matG_o = tl.load(matG_o_ptr)  # (siz_B, DH)

        ## load the c_t, n_t, m_t states for the current time step idx_t from idx_t+1
        # (states_all contains the initial states at idx_t=0)
        matC_t_ptr = tl.make_block_ptr(
            base=states_all
            + idx_b_NH * str_matStatesAll_NH
            + (idx_t + 1) * str_matStatesAll_T
            + 1 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_b_B * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        matC_t = tl.load(matC_t_ptr)  # (siz_B, DH)

        matN_t_ptr = tl.make_block_ptr(
            base=states_all
            + idx_b_NH * str_matStatesAll_NH
            + (idx_t + 1) * str_matStatesAll_T
            + 2 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_b_B * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        matN_t = tl.load(matN_t_ptr)  # (siz_B, DH)

        matM_t_ptr = tl.make_block_ptr(
            base=states_all
            + idx_b_NH * str_matStatesAll_NH
            + (idx_t + 1) * str_matStatesAll_T
            + 3 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_b_B * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        matM_t = tl.load(matM_t_ptr).to(tl.float32)  # (siz_B, DH)

        ## load the h_t-1, c_t-1, n_t-1, m_t-1 states for the previous time step idx_t-1 from idx_t
        matH_tminus1_ptr = tl.make_block_ptr(
            base=states_all
            + idx_b_NH * str_matStatesAll_NH
            + (idx_t) * str_matStatesAll_T
            + 0 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_b_B * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        matH_tminus1 = tl.load(matH_tminus1_ptr)  # (siz_B, DH)

        matC_tminus1_ptr = tl.make_block_ptr(
            base=states_all
            + idx_b_NH * str_matStatesAll_NH
            + (idx_t) * str_matStatesAll_T
            + 1 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_b_B * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        matC_tminus1 = tl.load(matC_tminus1_ptr)  # (siz_B, DH)

        matN_tminus1_ptr = tl.make_block_ptr(
            base=states_all
            + idx_b_NH * str_matStatesAll_NH
            + (idx_t) * str_matStatesAll_T
            + 2 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_b_B * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        matN_tminus1 = tl.load(matN_tminus1_ptr)  # (siz_B, DH)

        matM_tminus1_ptr = tl.make_block_ptr(
            base=states_all
            + idx_b_NH * str_matStatesAll_NH
            + (idx_t) * str_matStatesAll_T
            + 3 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_b_B * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        matM_tminus1 = tl.load(matM_tminus1_ptr).to(tl.float32)  # (siz_B, DH)

        ## load the delta errors delta_h_t, delta_c_t, delta_n_t of the states from outside for the current time step idx_t
        matDeltaHtrans_out_t_ptr = tl.make_block_ptr(
            base=delta_states_all_outside
            + idx_b_NH * str_delta_states_all_outside_NH
            + idx_t * str_delta_states_all_outside_T
            + 0 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_b_B * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        matDeltaHtrans_out_t = tl.load(matDeltaHtrans_out_t_ptr)  # (siz_B, DH)

        matDeltaCtrans_out_t_ptr = tl.make_block_ptr(
            base=delta_states_all_outside
            + idx_b_NH * str_delta_states_all_outside_NH
            + idx_t * str_delta_states_all_outside_T
            + 1 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_b_B * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        matDeltaCtrans_out_t = tl.load(matDeltaCtrans_out_t_ptr)  # (siz_B, DH)

        matDeltaNtrans_out_t_ptr = tl.make_block_ptr(
            base=delta_states_all_outside
            + idx_b_NH * str_delta_states_all_outside_NH
            + idx_t * str_delta_states_all_outside_T
            + 1 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_b_B * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        matDeltaNtrans_out_t = tl.load(matDeltaNtrans_out_t_ptr)  # (siz_B, DH)

        ## compute the backward pointwise operations
        matDeltaH_t = matDeltaHtrans_out_t + matDeltaH_tplus1  # (siz_B, DH)
        matDeltaC_t = matDeltaCtrans_out_t + matDeltaC_tplus1  # (siz_B, DH)
        matDeltaN_t = matDeltaNtrans_out_t + matDeltaN_tplus1  # (siz_B, DH)

        # slstm pointwise backward
        matDeltaC_t = matDeltaC_t + matDeltaH_t * (matG_o / matN_t)  # (siz_B, DH)
        matDeltaN_t = matDeltaN_t - matDeltaH_t * (
            matG_o * matC_t / (matN_t * matN_t)
        )  # (siz_B, DH)

        matG_i = tl.exp(matG_ibar - matM_t)  # (siz_B, DH)
        matG_logfplusm = matM_tminus1 + tl.log(tl.sigmoid(matG_fbar))  # (siz_B, DH)
        matG_f = tl.exp(matG_logfplusm - matM_t)  # (siz_B, DH)

        # compute the delta gate errors # (siz_B, DHout)
        matDeltaGI = (matDeltaC_t * matG_z + matDeltaN_t) * matG_i
        matDeltaGF = (matDeltaC_t * matC_tminus1 + matDeltaN_t * matN_tminus1) * matG_f * tl.sigmoid(-matG_fbar)
        matDeltaGZ = matDeltaC_t * matG_i * (1 - matG_z * matG_z)
        matDeltaGO = matDeltaH_t * (matC_t / matN_t) * (1 - matG_o) * matG_o

        # compute the delta errors to previous states
        matDeltaC_tminus1 = matDeltaC_t * matG_f
        matDeltaN_tminus1 = matDeltaN_t * matG_f

        matDeltaH_tminus1 = tl.dot(matDeltaGI.to(DTYPE), matR_i)  # (siz_B, DHin)
        matDeltaH_tminus1 += tl.dot(matDeltaGF.to(DTYPE), matR_f)  # (siz_B, DHin)
        matDeltaH_tminus1 += tl.dot(matDeltaGZ.to(DTYPE), matR_z)  # (siz_B, DHin)
        matDeltaH_tminus1 += tl.dot(matDeltaGO.to(DTYPE), matR_o)  # (siz_B, DHin)

        # compute the delta errors to the recurrent weights
        # (DHout, DHin) = (DHout, DHin) + (DHout, siz_B) * (siz_B, DHin)
        matDeltaR_i += tl.dot(
            tl.trans(matDeltaGI.to(DTYPE)), matH_tminus1
        )  # (DHout, DHin)
        matDeltaR_f += tl.dot(
            tl.trans(matDeltaGF.to(DTYPE)), matH_tminus1
        )  # (DHout, DHin)
        matDeltaR_z += tl.dot(
            tl.trans(matDeltaGZ.to(DTYPE)), matH_tminus1
        )  # (DHout, DHin)
        matDeltaR_o += tl.dot(
            tl.trans(matDeltaGO.to(DTYPE)), matH_tminus1
        )  # (DHout, DHin)

        # compute the delta errors to the biases
        vecDeltaB_i += tl.sum(matDeltaGI, axis=0)  # (DH,)
        vecDeltaB_f += tl.sum(matDeltaGF, axis=0)  # (DH,)
        vecDeltaB_z += tl.sum(matDeltaGZ, axis=0)  # (DH,)
        vecDeltaB_o += tl.sum(matDeltaGO, axis=0)  # (DH,)

        ## store the deltaGate errors
        matDeltaGI_ptr = tl.make_block_ptr(
            base=delta_Wx
            + idx_b_NH * str_matDeltaWx_NH
            + idx_t * str_matDeltaWx_T
            + 0 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_b_B * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        tl.store(matDeltaGI_ptr, matDeltaGI.to(DTYPE))

        matDeltaGF_ptr = tl.make_block_ptr(
            base=delta_Wx
            + idx_b_NH * str_matDeltaWx_NH
            + idx_t * str_matDeltaWx_T
            + 1 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_b_B * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        tl.store(matDeltaGF_ptr, matDeltaGF.to(DTYPE))

        matDeltaGZ_ptr = tl.make_block_ptr(
            base=delta_Wx
            + idx_b_NH * str_matDeltaWx_NH
            + idx_t * str_matDeltaWx_T
            + 2 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_b_B * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        tl.store(matDeltaGZ_ptr, matDeltaGZ.to(DTYPE))

        matDeltaGO_ptr = tl.make_block_ptr(
            base=delta_Wx
            + idx_b_NH * str_matDeltaWx_NH
            + idx_t * str_matDeltaWx_T
            + 3 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_b_B * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        tl.store(matDeltaGO_ptr, matDeltaGO.to(DTYPE))

        ## next iteration
        matDeltaH_tplus1 = matDeltaH_tminus1  # (siz_B, DH)
        matDeltaC_tplus1 = matDeltaC_tminus1  # (siz_B, DH)
        matDeltaN_tplus1 = matDeltaN_tminus1  # (siz_B, DH)

    ## store the delta errors to the initial states
    matDeltaHtrans_initial_ptr = tl.make_block_ptr(
        base=delta_states_initial + idx_b_NH * NS * B * DH + 0 * B * DH,
        shape=(B, DH),
        strides=(DH, 1),
        offsets=(idx_b_B * siz_B, 0),
        block_shape=(siz_B, DH),
        order=(0, 1),
    )
    tl.store(matDeltaHtrans_initial_ptr, matDeltaH_tplus1.to(DTYPE))

    matDeltaCtrans_initial_ptr = tl.make_block_ptr(
        base=delta_states_initial + idx_b_NH * NS * B * DH + 1 * B * DH,
        shape=(B, DH),
        strides=(DH, 1),
        offsets=(idx_b_B * siz_B, 0),
        block_shape=(siz_B, DH),
        order=(0, 1),
    )
    tl.store(matDeltaCtrans_initial_ptr, matDeltaC_tplus1.to(DTYPE))

    matDeltaNtrans_initial_ptr = tl.make_block_ptr(
        base=delta_states_initial + idx_b_NH * NS * B * DH + 2 * B * DH,
        shape=(B, DH),
        strides=(DH, 1),
        offsets=(idx_b_B * siz_B, 0),
        block_shape=(siz_B, DH),
        order=(0, 1),
    )
    tl.store(matDeltaNtrans_initial_ptr, matDeltaN_tplus1.to(DTYPE))

    # set the initial m state gradient to zero
    matDeltaMtrans_initial_ptr = tl.make_block_ptr(
        base=delta_states_initial + idx_b_NH * NS * B * DH + 3 * B * DH,
        shape=(B, DH),
        strides=(DH, 1),
        offsets=(idx_b_B * siz_B, 0),
        block_shape=(siz_B, DH),
        order=(0, 1),
    )
    tl.store(matDeltaMtrans_initial_ptr, tl.zeros((siz_B, DH), dtype=DTYPE))

    ## store the delta errors to the recurrent weights
    matDeltaR_i_ptr = tl.make_block_ptr(
        base=delta_R + idx_b_B * str_matR_B + idx_b_NH * str_matR_NH + 0 * str_matR_NGR,
        shape=(DH, DH),
        strides=(DH, 1),
        offsets=(0, 0),
        block_shape=(DH, DH),
        order=(0, 1),
    )
    tl.store(matDeltaR_i_ptr, matDeltaR_i.to(DTYPE))

    matDeltaR_f_ptr = tl.make_block_ptr(
        base=delta_R + idx_b_B * str_matR_B + idx_b_NH * str_matR_NH + 1 * str_matR_NGR,
        shape=(DH, DH),
        strides=(DH, 1),
        offsets=(0, 0),
        block_shape=(DH, DH),
        order=(0, 1),
    )
    tl.store(matDeltaR_f_ptr, matDeltaR_f.to(DTYPE))

    matDeltaR_z_ptr = tl.make_block_ptr(
        base=delta_R + idx_b_B * str_matR_B + idx_b_NH * str_matR_NH + 2 * str_matR_NGR,
        shape=(DH, DH),
        strides=(DH, 1),
        offsets=(0, 0),
        block_shape=(DH, DH),
        order=(0, 1),
    )
    tl.store(matDeltaR_z_ptr, matDeltaR_z.to(DTYPE))

    matDeltaR_o_ptr = tl.make_block_ptr(
        base=delta_R + idx_b_B * str_matR_B + idx_b_NH * str_matR_NH + 3 * str_matR_NGR,
        shape=(DH, DH),
        strides=(DH, 1),
        offsets=(0, 0),
        block_shape=(DH, DH),
        order=(0, 1),
    )
    tl.store(matDeltaR_o_ptr, matDeltaR_o.to(DTYPE))

    ## store the delta errors to the biases
    vecDeltaB_i_ptr = (
        delta_b
        + idx_b_B * NH * NGI * DH
        + idx_b_NH * NGI * DH
        + 0 * DH
        + tl.arange(0, DH)
    )
    tl.store(vecDeltaB_i_ptr, vecDeltaB_i.to(DTYPE))

    vecDeltaB_f_ptr = (
        delta_b
        + idx_b_B * NH * NGI * DH
        + idx_b_NH * NGI * DH
        + 1 * DH
        + tl.arange(0, DH)
    )
    tl.store(vecDeltaB_f_ptr, vecDeltaB_f.to(DTYPE))

    vecDeltaB_z_ptr = (
        delta_b
        + idx_b_B * NH * NGI * DH
        + idx_b_NH * NGI * DH
        + 2 * DH
        + tl.arange(0, DH)
    )
    tl.store(vecDeltaB_z_ptr, vecDeltaB_z.to(DTYPE))

    vecDeltaB_o_ptr = (
        delta_b
        + idx_b_B * NH * NGI * DH
        + idx_b_NH * NGI * DH
        + 3 * DH
        + tl.arange(0, DH)
    )
    tl.store(vecDeltaB_o_ptr, vecDeltaB_o.to(DTYPE))


def backward_sequence(
    delta_states_all_outside: torch.Tensor,  # (T, NS, B, NH, D) delta errors from all states
    delta_states_last_outside: torch.Tensor,  # (NS, B, NH, D) delta errors from the last state
    # Wx: torch.Tensor,  # (B, T, NGI, NH, D) inputs
    R: torch.Tensor,  # (NGR, NH, Dout, Din) recurrent weights (Dout == Din == D)
    # b: torch.Tensor,  # (NGI, NH, D) biases
    states_all: torch.Tensor,  # (T+1, NS, B, NH, D) all states
    gates_all: torch.Tensor,  # (T, NGI, B, NH, D) all gates
    backward_recurrent_clip_val: float | None = None,
    siz_B: int = 16,  # the number of batches per thread block
    true_B: int = None,  # true batch size, we add this in order to avoid removing and adding the padding during forward and backward
) -> tuple[
    torch.Tensor,  # (NS, B, NH, D) delta errors to initial states
    torch.Tensor,  # (B, T, NGI, NH, D) delta errors to inputs
    torch.Tensor,  # (NGR, NH, Dout, Din) delta errors to recurrent weights
    torch.Tensor,  # (NH, NGI, D) delta errors to biases
]:
    # support the case where delta_states_last_outside has the time dimension explicitly
    T_dim_explicit = False
    if delta_states_last_outside.ndim == 5:
        assert delta_states_last_outside.shape[0] == 1, f"states_initial.shape[0] must be 1: got {delta_states_last_outside.shape}."
        T_dim_explicit = True
        delta_states_last_outside = delta_states_last_outside[0]

    if true_B is None:
        true_B = delta_states_all_outside.shape[2]
    T, NS, _, NH, DH = delta_states_all_outside.shape
    NGR, _, _, _ = R.shape
    assert NS == 4, "sLSTM has 4 states: h, c, n, m."
    assert NGR == 4, "sLSTM has 4 gates: i, f, z, o."
    NGI = NGR
    DHout, DHin = DH, DH
    dtype = R.dtype
    device = R.device

    assert R.shape[1:] == (NH, DH, DH)

    assert (
        delta_states_all_outside.dtype == dtype
    ), f"dtype mismatch: delta_states_all.dtype: {R.dtype}, R.dtype: {dtype}."
    assert (
        delta_states_last_outside.dtype == dtype
    ), f"dtype mismatch: delta_states_last.dtype: {R.dtype}, R.dtype: {dtype}."

    assert is_power_of_2(DH), f"head dimension must be a power of 2, got {DH}."

    MIN_BATCH_SIZE = 16  # we need at least 16 batches for tl.dot() (16x16 tensor cores)
    ## batch size padding to be a multiple of MIN_BATCH_SIZE
    effective_B = next_multiple_of(true_B, MIN_BATCH_SIZE)
    if effective_B != true_B:
        delta_states_all_outside = torch.cat(
            [
                delta_states_all_outside,
                torch.zeros(
                    [T, NS, effective_B - true_B, NH, DH], dtype=dtype, device=device
                ),
            ],
            dim=2,
        )
        delta_states_last_outside = torch.cat(
            [
                delta_states_last_outside,
                torch.zeros(
                    [NS, effective_B - true_B, NH, DH], dtype=dtype, device=device
                ),
            ],
            dim=1,
        )

    # Reshapes for kernel
    # we always want the number of heads to be the first dimension
    # as we parallelize along this dimensoin
    R_kshaped = rearrange(R, "ngr nh dout din -> nh ngr dout din").contiguous()

    delta_states_all_outside_kshaped = rearrange(
        delta_states_all_outside, "t ns b nh dh -> nh t ns b dh"
    ).contiguous()
    delta_states_last_outside_kshaped = rearrange(
        delta_states_last_outside, "ns b nh dh -> nh ns b dh"
    ).contiguous()

    states_all_kshaped = rearrange(
        states_all, "t ns b nh dh -> nh t ns b dh"
    ).contiguous()
    gates_all_kshaped = rearrange(
        gates_all, "t ngi b nh dh -> nh t ngi b dh"
    ).contiguous()

    # kernel call
    num_B = triton.cdiv(effective_B, siz_B)
    grid = (NH, num_B)

    # Allocate output tensors
    delta_Wx = torch.empty([NH, T, NGI, effective_B, DH], dtype=dtype, device=device)
    delta_states_initial = torch.empty(
        [NH, NS, effective_B, DH], dtype=dtype, device=device
    )
    # we need to add the batch dimension grid dimension to the output shape
    # we sum them up outside the kernel
    delta_R = torch.empty([num_B, NH, NGR, DHout, DHin], dtype=dtype, device=device)
    delta_b = torch.empty([num_B, NH, NGI, DH], dtype=dtype, device=device)

    if backward_recurrent_clip_val is None:
        backward_recurrent_clip_val = -1.0

    _backward_sequence_kernel[grid](
        delta_states_all_outside=delta_states_all_outside_kshaped,
        delta_states_last_outside=delta_states_last_outside_kshaped,
        R=R_kshaped,
        states_all=states_all_kshaped,
        gates_all=gates_all_kshaped,
        delta_states_initial=delta_states_initial,
        delta_Wx=delta_Wx,
        delta_R=delta_R,
        delta_b=delta_b,
        T=T,
        NS=NS,
        B=effective_B,
        NH=NH,
        DH=DH,
        NGI=NGI,
        NGR=NGR,
        siz_B=siz_B,
        DTYPE=torch2triton_dtype(dtype),
        backward_recurrent_clip_val=backward_recurrent_clip_val,
        num_warps=4,
    )

    delta_R = delta_R.sum(0)
    delta_b = delta_b.sum(0)

    delta_R = rearrange(delta_R, "nh ngr dout din -> ngr nh dout din")
    delta_Wx = rearrange(delta_Wx, "nh t ngi b dh -> b t ngi nh dh")
    delta_b = rearrange(delta_b, "nh ngi dh -> ngi nh dh")
    delta_states_initial = rearrange(delta_states_initial, "nh ns b dh -> ns b nh dh")
    ## batch_size padding
    delta_states_initial = delta_states_initial[:, :true_B, ...]
    delta_Wx = delta_Wx[:true_B, ...]

    if T_dim_explicit:
        delta_states_initial = delta_states_initial.unsqueeze(0)

    return delta_states_initial, delta_Wx, delta_R, delta_b
