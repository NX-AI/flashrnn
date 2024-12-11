# Maximilian Beck
import torch
import triton
import triton.language as tl
from einops import rearrange
from triton import OutOfResources

from .triton_utils import is_power_of_2, next_multiple_of, torch2triton_dtype

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

ENABLE_AUTOTUNING = False

if ENABLE_AUTOTUNING:
    configs = [
        triton.Config({"siz_B": siz_B}, num_stages=s, num_warps=w)
        for siz_B in [16, 32, 64]
        for s in [1]
        for w in [1, 2, 4, 8]
    ]
else:
    configs = [
        triton.Config({"siz_B": siz_B}, num_stages=s, num_warps=w)
        for siz_B in [16]
        for s in [1]
        for w in [4]
    ]


@triton.jit
def triton_tanh(x):
    return (1.0 - tl.exp(-2.0 * x)) / (1.0 + tl.exp(-2.0 * x))


@triton.autotune(configs, key=["siz_B", "T", "B", "NH", "DH"])
@triton.jit
def _forward_sequence_kernel(
    states_initial,  # (NH, NS, B, DH) (order: h c n m)
    Wx,  # (NH, T, DGI, B, DH) (order: i f z o)
    R,  # (NH, NGR, DHin, DHout)
    b,  # (NH, NGI, DH)
    states_all,  # (NH, T, NS, B, DH)
    gates_all,  # (NH, T, NGI, B, DH)
    # dimensions,
    T: tl.constexpr,  # sequence length
    NS: tl.constexpr,  # number of states
    B: tl.constexpr,  # batch size
    NH: tl.constexpr,  # number of heads
    DH: tl.constexpr,  # head dimension
    NGI: tl.constexpr,  # number of gates that depend on input
    NGR: tl.constexpr,  # number of gates that depend on recurrent state
    siz_B: tl.constexpr,  # the number of batches per threadblock
    OUTPUT_GATES: tl.constexpr,
    DTYPE: tl.constexpr = tl.float32,
):
    idx_b_NH, idx_b_B = tl.program_id(0), tl.program_id(1)

    ## compute the strides
    str_matWx_NH = T * NGI * B * DH
    str_matWx_T = NGI * B * DH
    str_matStatesAll_NH = (T + 1) * NS * B * DH
    str_matStatesAll_T = NS * B * DH
    str_matGatesAll_NH = T * NGI * B * DH
    str_matGatesAll_T = NGI * B * DH
    ##

    ## load the initial states
    # load initial h state to be used for the recurrence
    matHtrans_initial_ptr = tl.make_block_ptr(
        base=states_initial + idx_b_NH * NS * B * DH + 0 * B * DH,
        shape=(B, DH),
        strides=(DH, 1),
        offsets=(idx_b_B * siz_B, 0),
        block_shape=(siz_B, DH),
        order=(0, 1),
    )
    matHtrans = tl.load(matHtrans_initial_ptr).to(tl.float32)  # (B, DH)

    # load initial c state
    matCtrans_initial_ptr = tl.make_block_ptr(
        base=states_initial + idx_b_NH * NS * B * DH + 1 * B * DH,
        shape=(B, DH),
        strides=(DH, 1),
        offsets=(idx_b_B * siz_B, 0),
        block_shape=(siz_B, DH),
        order=(0, 1),
    )
    matCtrans = tl.load(matCtrans_initial_ptr).to(tl.float32)  # (B, DH)

    # load initial n state
    matNtrans_initial_ptr = tl.make_block_ptr(
        base=states_initial + idx_b_NH * NS * B * DH + 2 * B * DH,
        shape=(B, DH),
        strides=(DH, 1),
        offsets=(idx_b_B * siz_B, 0),
        block_shape=(siz_B, DH),
        order=(0, 1),
    )
    matNtrans = tl.load(matNtrans_initial_ptr).to(tl.float32)  # (B, DH)

    # load initial m state
    matMtrans_initial_ptr = tl.make_block_ptr(
        base=states_initial + idx_b_NH * NS * B * DH + 3 * B * DH,
        shape=(B, DH),
        strides=(DH, 1),
        offsets=(idx_b_B * siz_B, 0),
        block_shape=(siz_B, DH),
        order=(0, 1),
    )
    matMtrans = tl.load(matMtrans_initial_ptr).to(tl.float32)  # (B, DH)

    ## store initial states
    # store initial h state in states all
    matHtrans_initial_store_ptr = tl.make_block_ptr(
        base=states_all
        + idx_b_NH * str_matStatesAll_NH
        + 0 * str_matStatesAll_T
        + 0 * B * DH,
        shape=(B, DH),
        strides=(DH, 1),
        offsets=(idx_b_B * siz_B, 0),
        block_shape=(siz_B, DH),
        order=(0, 1),
    )
    tl.store(matHtrans_initial_store_ptr, matHtrans.to(DTYPE))
    # store initial c state in states all
    matCtrans_initial_store_ptr = tl.make_block_ptr(
        base=states_all
        + idx_b_NH * str_matStatesAll_NH
        + 0 * str_matStatesAll_T
        + 1 * B * DH,
        shape=(B, DH),
        strides=(DH, 1),
        offsets=(idx_b_B * siz_B, 0),
        block_shape=(siz_B, DH),
        order=(0, 1),
    )
    tl.store(matCtrans_initial_store_ptr, matCtrans.to(DTYPE))
    # store initial n state in states all
    matNtrans_initial_store_ptr = tl.make_block_ptr(
        base=states_all
        + idx_b_NH * str_matStatesAll_NH
        + 0 * str_matStatesAll_T
        + 2 * B * DH,
        shape=(B, DH),
        strides=(DH, 1),
        offsets=(idx_b_B * siz_B, 0),
        block_shape=(siz_B, DH),
        order=(0, 1),
    )
    tl.store(matNtrans_initial_store_ptr, matNtrans.to(DTYPE))
    # store initial m state in states all
    matMtrans_initial_store_ptr = tl.make_block_ptr(
        base=states_all
        + idx_b_NH * str_matStatesAll_NH
        + 0 * str_matStatesAll_T
        + 3 * B * DH,
        shape=(B, DH),
        strides=(DH, 1),
        offsets=(idx_b_B * siz_B, 0),
        block_shape=(siz_B, DH),
        order=(0, 1),
    )
    tl.store(matMtrans_initial_store_ptr, matMtrans.to(DTYPE))

    ## load recurrent weights only once
    # load the recurrent weights
    matRtrans_i_ptr = tl.make_block_ptr(
        base=R + idx_b_NH * DH * NGR * DH + 0 * DH * DH,
        shape=(DH, DH),
        strides=(DH, 1),
        offsets=(0, 0),
        block_shape=(DH, DH),
        order=(0, 1),
    )
    matRtrans_i = tl.load(matRtrans_i_ptr)  # (DHin, DHout)

    matRtrans_f_ptr = tl.make_block_ptr(
        base=R + idx_b_NH * DH * NGR * DH + 1 * DH * DH,
        shape=(DH, DH),
        strides=(DH, 1),
        offsets=(0, 0),
        block_shape=(DH, DH),
        order=(0, 1),
    )
    matRtrans_f = tl.load(matRtrans_f_ptr)  # (DHin, DHout)

    matRtrans_z_ptr = tl.make_block_ptr(
        base=R + idx_b_NH * DH * NGR * DH + 2 * DH * DH,
        shape=(DH, DH),
        strides=(DH, 1),
        offsets=(0, 0),
        block_shape=(DH, DH),
        order=(0, 1),
    )
    matRtrans_z = tl.load(matRtrans_z_ptr)  # (DHin, DHout)

    matRtrans_o_ptr = tl.make_block_ptr(
        base=R + idx_b_NH * DH * NGR * DH + 3 * DH * DH,
        shape=(DH, DH),
        strides=(DH, 1),
        offsets=(0, 0),
        block_shape=(DH, DH),
        order=(0, 1),
    )
    matRtrans_o = tl.load(matRtrans_o_ptr)  # (DHin, DHout)

    ## load the biases only once
    vecB_i_ptr = b + idx_b_NH * NGI * DH + 0 * DH + tl.arange(0, DH)
    vecB_i = tl.load(vecB_i_ptr)  # (DH,)

    vecB_f_ptr = b + idx_b_NH * NGI * DH + 1 * DH + tl.arange(0, DH)
    vecB_f = tl.load(vecB_f_ptr)  # (DH,)

    vecB_z_ptr = b + idx_b_NH * NGI * DH + 2 * DH + tl.arange(0, DH)
    vecB_z = tl.load(vecB_z_ptr)  # (DH,)

    vecB_o_ptr = b + idx_b_NH * NGI * DH + 3 * DH + tl.arange(0, DH)
    vecB_o = tl.load(vecB_o_ptr)  # (DH,)

    for idx_t in range(T):
        ## load gate preactivations Wx per time step
        matIxtrans_ptr = tl.make_block_ptr(
            base=Wx + idx_b_NH * str_matWx_NH + idx_t * str_matWx_T + 0 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_b_B * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        matIxtrans = tl.load(matIxtrans_ptr)  # (B, DH)

        matFxtrans_ptr = tl.make_block_ptr(
            base=Wx + idx_b_NH * str_matWx_NH + idx_t * str_matWx_T + 1 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_b_B * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        matFxtrans = tl.load(matFxtrans_ptr)  # (B, DH)

        matZxtrans_ptr = tl.make_block_ptr(
            base=Wx + idx_b_NH * str_matWx_NH + idx_t * str_matWx_T + 2 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_b_B * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        matZxtrans = tl.load(matZxtrans_ptr)  # (B, DH)

        matOxtrans_ptr = tl.make_block_ptr(
            base=Wx + idx_b_NH * str_matWx_NH + idx_t * str_matWx_T + 3 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_b_B * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        matOxtrans = tl.load(matOxtrans_ptr)  # (B, DH)

        ## compute recurrent gate preactivations (matrix multiplication)
        matRhtrans_i = tl.dot(matHtrans.to(DTYPE), matRtrans_i)  # (B, DH)
        matRhtrans_f = tl.dot(matHtrans.to(DTYPE), matRtrans_f)  # (B, DH)
        matRhtrans_z = tl.dot(matHtrans.to(DTYPE), matRtrans_z)  # (B, DH)
        matRhtrans_o = tl.dot(matHtrans.to(DTYPE), matRtrans_o)  # (B, DH)

        ## compute the gate preactivations
        matIbar = matIxtrans + matRhtrans_i + vecB_i[None, :]  # (B, DH)
        matFbar = matFxtrans + matRhtrans_f + vecB_f[None, :]  # (B, DH)
        matZbar = matZxtrans + matRhtrans_z + vecB_z[None, :]  # (B, DH)
        matObar = matOxtrans + matRhtrans_o + vecB_o[None, :]  # (B, DH)

        ## compute the pointwise operations
        matLogFplusM = matMtrans + tl.log(tl.sigmoid(matFbar))

        matMtrans_next = tl.where(
            matNtrans == 0.0, matIbar, tl.maximum(matIbar, matLogFplusM)
        )

        matI = tl.exp(matIbar - matMtrans_next)
        matF = tl.exp(matLogFplusM - matMtrans_next)
        matZ = triton_tanh(matZbar)
        matO = tl.sigmoid(matObar)

        ## memory cell updates
        matCtrans_next = matF * matCtrans + matI * matZ
        ## > 1 by construction, maybe not numerically sometimes
        matNtrans_next = tl.maximum(matF * matNtrans + matI, 1.0)

        matHtrans_next = matO * (matCtrans_next / matNtrans_next)

        ## store the new states
        matHtrans_next_ptr = tl.make_block_ptr(
            base=states_all
            + idx_b_NH * str_matStatesAll_NH
            + (idx_t + 1) * str_matStatesAll_T
            + 0 * B * DH,
            shape=(B, DH),
            strides=(DH, 1),
            offsets=(idx_b_B * siz_B, 0),
            block_shape=(siz_B, DH),
            order=(0, 1),
        )
        tl.store(matHtrans_next_ptr, matHtrans_next.to(DTYPE))

        matCtrans_next_ptr = tl.make_block_ptr(
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
        tl.store(matCtrans_next_ptr, matCtrans_next.to(DTYPE))

        matNtrans_next_ptr = tl.make_block_ptr(
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
        tl.store(matNtrans_next_ptr, matNtrans_next.to(DTYPE))

        matMtrans_next_ptr = tl.make_block_ptr(
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
        tl.store(matMtrans_next_ptr, matMtrans_next.to(DTYPE))

        ## [optional] store the gates
        if OUTPUT_GATES:
            matGatesItrans_ptr = tl.make_block_ptr(
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
            tl.store(matGatesItrans_ptr, matIbar.to(DTYPE))

            matGatesFtrans_ptr = tl.make_block_ptr(
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
            tl.store(matGatesFtrans_ptr, matFbar.to(DTYPE))

            matGatesZtrans_ptr = tl.make_block_ptr(
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
            tl.store(matGatesZtrans_ptr, matZ.to(DTYPE))

            matGatesOtrans_ptr = tl.make_block_ptr(
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
            tl.store(matGatesOtrans_ptr, matO.to(DTYPE))

        ## move the states to the next time step
        matCtrans = matCtrans_next
        matHtrans = matHtrans_next
        matNtrans = matNtrans_next
        matMtrans = matMtrans_next


def forward_sequence(
    states_initial: torch.Tensor,  # (NS, B, NH, DH) initial states (h state, c state etc. (h state is used for recurrent weights and is always first))
    Wx: torch.Tensor,  # (B, T, NGI, NH, DH) inputs
    R: torch.Tensor,  # (NGR, NH, Dout, Din) recurrent weights (Dout == Din == D)
    b: torch.Tensor,  # (NGI, NGI, DH) biases
    output_gates_and_states_initial: bool = False,
    # siz_B: int = 16,  # the number of batches per threadblock
) -> (
    tuple[
        torch.Tensor,  # (T, NS, B, NH, D) all states
        torch.Tensor,  # (NS, B, NH, D) last state
    ]
    | tuple[
        tuple[
            torch.Tensor,  # (T, NS, B, NH, D) all states
            torch.Tensor,  # (NS, B, NH, D) last state
        ],
        torch.Tensor,  # (T, NGI, B, NH, D) gates
    ]
):
    # support the case where states initial has the time dimension explicitly
    T_dim_explicit = False
    if states_initial.ndim == 5:
        assert (
            states_initial.shape[0] == 1
        ), f"states_initial.shape[0] must be 1: got {states_initial.shape}."
        T_dim_explicit = True
        states_initial = states_initial[0]
    NS, B, NH, DH = states_initial.shape
    _, T, NGI, _, _ = Wx.shape
    NGR, _, _, _ = R.shape
    assert R.shape[1:] == (NH, DH, DH)
    assert b.shape == (NGI, NH, DH)
    assert NS == 4, "sLSTM has 4 states: h, c, n, m."
    assert NGR == 4, "sLSTM has 4 gates: i, f, z, o."
    assert NGR == NGI, "NGR must be equal to NGI."

    dtype = Wx.dtype
    device = Wx.device

    assert R.dtype == dtype, f"dtype mismatch: R.dtype: {R.dtype}, Wx.dtype: {dtype}."

    assert is_power_of_2(DH), f"DH must be a power of 2, got {DH}."
    MIN_BATCH_SIZE = 16  # we need at least 16 batches for tl.dot() (16x16 tensor cores)
    ## batch size padding to be a multiple of MIN_BATCH_SIZE
    effective_B = next_multiple_of(B, MIN_BATCH_SIZE)
    if effective_B != B:
        states_initial = torch.cat(
            [
                states_initial,
                torch.zeros([NS, effective_B - B, NH, DH], device=device, dtype=dtype),
            ],
            dim=1,
        )
        Wx = torch.cat(
            [
                Wx,
                torch.zeros(
                    [effective_B - B, T, NGI, NH, DH], device=device, dtype=dtype
                ),
            ],
            dim=0,
        )
    ## end of batch size padding

    states_all = torch.empty(
        [NH, T + 1, NS, effective_B, DH], device=device, dtype=dtype
    )

    if output_gates_and_states_initial:
        gates_all = torch.empty(
            [NH, T, NGI, effective_B, DH], device=device, dtype=dtype
        )
    else:
        gates_all = None

    # reshape the inputs for the kernel (they must be contiguous)
    states_initial_kshaped = rearrange(
        states_initial, "ns b nh dh -> nh ns b dh"
    ).contiguous()

    Wx_kshaped = rearrange(Wx, "b t ngi nh dh -> nh t ngi b dh").contiguous()
    R_kshaped = rearrange(R, "ngr nh dhout dhin -> nh ngr dhin dhout").contiguous()
    b_kshaped = rearrange(b, "ngi nh dh -> nh ngi dh").contiguous()
    # call the kernel
    # num_b_B = triton.cdiv(B, siz_B)

    def grid(args):
        siz_B = args["siz_B"]
        assert siz_B >= MIN_BATCH_SIZE, "siz_B must be at least 16"
        if siz_B > effective_B:
            # we raise this to skip it in the autotuning
            raise OutOfResources(required=siz_B, limit=effective_B, name="siz_B")
            # raise CompileTimeAssertionFailure("siz_B must be at most B")
        g = (NH, triton.cdiv(B, siz_B))
        return g

    # grid = (NH, num_b_B)
    _forward_sequence_kernel[grid](
        states_initial=states_initial_kshaped,
        Wx=Wx_kshaped,
        R=R_kshaped,
        b=b_kshaped,
        states_all=states_all,
        gates_all=gates_all,
        T=T,
        NS=NS,
        B=effective_B,
        NH=NH,
        DH=DH,
        NGI=NGI,
        NGR=NGR,
        # siz_B=siz_B,
        OUTPUT_GATES=output_gates_and_states_initial,
        DTYPE=torch2triton_dtype(dtype),
    )

    states_out = rearrange(states_all, "nh t ns b dh -> t ns b nh dh", ns=NS, dh=DH)

    if output_gates_and_states_initial:
        # we do not remove the padding here as we assume we call backward after forward
        # gates_all = gates_all[:, :, :, :B, :]
        gates_out = rearrange(
            gates_all, "nh t ngi b dh -> t ngi b nh dh", ngi=NGI, dh=DH
        )
        if T_dim_explicit:
            return (states_out, states_out[-1:]), gates_out
        return (states_out, states_out[-1]), gates_out
    else:
        ## remove the padding only when not outputting the gates for backward
        states_out = states_out[:, :, :B, :, :]
        ##
        if T_dim_explicit:
            return states_out[1:], states_out[-1:]
        return states_out[1:], states_out[-1]
