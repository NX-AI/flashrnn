# Maximilian Beck

import torch
from typing import Callable

from torch.amp import custom_fwd, custom_bwd

from .fw import forward_sequence, slstm_pointwise_fw, lstm_pointwise_fw
from .bw import backward_sequence, slstm_pointwise_bw, lstm_pointwise_bw

# Dimensions:
# B: batch size
# T: sequence length
# NGI: number of gates that depend on input
# NGR: number of gates that depend on recurrent state
# NH: number of heads
# D: hidden dimension
# NS: number of states

# Note: NS dimension
# > in NS dimension the first index is the state that is used for recurrent weights

# TODO should we pass the gates to the backward pass or should we recompute them?


def _rnn_fwbw_generator(
    autocast_kernel_dtype: torch.dtype, pointwise_fw: Callable, pointwise_bw: Callable
) -> torch.autograd.Function:
    class _rnn_fwbw(torch.autograd.Function):
        @staticmethod
        @custom_fwd(device_type="cuda", cast_inputs=autocast_kernel_dtype)
        def forward(
            ctx,
            states_initial: torch.Tensor,  # (NS, B, NH, D) initial states
            Wx: torch.Tensor,  # (B, T, NGI, NH, D) inputs
            R: torch.Tensor,  # (NGR, NH, Dout, Din) recurrent weights (Dout == Din == D)
            b: torch.Tensor,  # (NH, NGI, D) biases
            backward_recurrent_clip_val: float | None = None,
        ) -> tuple[
            torch.Tensor,  # (T, NS, B, NH, D) all states
            torch.Tensor,  # (NS, B, NH, D) last state
        ]:
            # all_states as (T+1, NS, B, NH, D) and last_state as (NS, B, NH, D)
            (all_states, last_state), all_gates = forward_sequence(
                states_initial=states_initial,
                Wx=Wx,
                R=R,
                b=b,
                forward_pointwise=pointwise_fw,
                output_gates_and_states_initial=True,
            )
            ctx.save_for_backward(
                all_states, all_gates, R, backward_recurrent_clip_val
            )
            return all_states[1:], last_state

        @staticmethod
        @custom_bwd(device_type="cuda")
        def backward(
            ctx,
            delta_states_all_outside: torch.Tensor,  # (T, NS, B, NH, D) all state delta errors from outside
            delta_states_last_outside: torch.Tensor,  # (NS, B, NH, D) last state delta errors from outside
        ) -> tuple[
            torch.Tensor,  # (NS, B, NH, D) delta errors to initial states
            torch.Tensor,  # (B, T, NGI, NH, D) delta errors to inputs
            torch.Tensor,  # (NGR, NH, Dout, Din) delta errors to recurrent weights
            torch.Tensor,  # (NH, NGI, D) delta errors to biases
        ]:
            (all_states, all_gates, R, backward_recurrent_clip_val) = (
                ctx.saved_tensors
            )

            delta_states_initial, delta_Wx, delta_R, delta_b = backward_sequence(
                delta_states_all_outside=delta_states_all_outside,
                delta_states_last_outside=delta_states_last_outside,
                R=R,
                states_all=all_states,
                gates_all=all_gates,
                backward_pointwise=pointwise_bw,
                backward_recurrent_clip_val=backward_recurrent_clip_val,
            )
            return delta_states_initial, delta_Wx, delta_R, delta_b, None

    return _rnn_fwbw


lstm_pt_fp32_fwbw = _rnn_fwbw_generator(
    torch.float32, lstm_pointwise_fw, lstm_pointwise_bw
)
lstm_pt_fp16_fwbw = _rnn_fwbw_generator(
    torch.float16, lstm_pointwise_fw, lstm_pointwise_bw
)
lstm_pt_bf16_fwbw = _rnn_fwbw_generator(
    torch.bfloat16, lstm_pointwise_fw, lstm_pointwise_bw
)

slstm_pt_fp32_fwbw = _rnn_fwbw_generator(
    torch.float32, slstm_pointwise_fw, slstm_pointwise_bw
)
slstm_pt_fp16_fwbw = _rnn_fwbw_generator(
    torch.float16, slstm_pointwise_fw, slstm_pointwise_bw
)
slstm_pt_bf16_fwbw = _rnn_fwbw_generator(
    torch.bfloat16, slstm_pointwise_fw, slstm_pointwise_bw
)

lstm_pt_registry = {
    "float32": lstm_pt_fp32_fwbw,
    "float16": lstm_pt_fp16_fwbw,
    "bfloat16": lstm_pt_bf16_fwbw,
}

slstm_pt_registry = {
    "float32": slstm_pt_fp32_fwbw,
    "float16": slstm_pt_fp16_fwbw,
    "bfloat16": slstm_pt_bf16_fwbw,
}


def lstm_pt_fwbw(
    states_initial: torch.Tensor,  # (NS, B, NH, D) initial states
    Wx: torch.Tensor,  # (B, T, NGI, NH, D) inputs
    R: torch.Tensor,  # (NGR, NH, Dout, Din) recurrent weights (Dout == Din == D)
    b: torch.Tensor,  # (NH, NGI, D) biases
    backward_recurrent_clip_val: float | None = None,
    autocast_kernel_dtype: str = "float32",
) -> tuple[
    torch.Tensor,  # (T, NS, B, NH, D) all states
    torch.Tensor,  # (NS, B, NH, D) last state
]:
    lstm_func = lstm_pt_registry[autocast_kernel_dtype]

    all_states, last_state = lstm_func.apply(
        states_initial, Wx, R, b, backward_recurrent_clip_val
    )
    return all_states, last_state


def slstm_pt_fwbw(
    states_initial: torch.Tensor,  # (NS, B, NH, D) initial states
    Wx: torch.Tensor,  # (B, T, NGI, NH, D) inputs
    R: torch.Tensor,  # (NGR, NH, Dout, Din) recurrent weights (Dout == Din == D)
    b: torch.Tensor,  # (NH, NGI, D) biases
    backward_recurrent_clip_val: float | None = None,
    autocast_kernel_dtype: str = "float32",
) -> tuple[
    torch.Tensor,  # (T, NS, B, NH, D) all states
    torch.Tensor,  # (NS, B, NH, D) last state
]:
    slstm_func = slstm_pt_registry[autocast_kernel_dtype]

    all_states, last_state = slstm_func.apply(
        states_initial, Wx, R, b, backward_recurrent_clip_val
    )
    return all_states, last_state
