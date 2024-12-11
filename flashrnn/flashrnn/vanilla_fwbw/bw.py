# Maximilian Beck
import torch
from einops import rearrange
from typing import Callable, Optional

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


def backward_sequence(
    delta_states_all_outside: torch.Tensor,  # (T, NS, B, NH, D) delta errors from all states
    delta_states_last_outside: torch.Tensor,  # (NS, B, NH, D) delta errors from the last state
    # Wx: torch.Tensor,  # (B, T, NGI, NH, D) inputs
    R: torch.Tensor,  # (NGR, NH, Dout, Din) recurrent weights (Dout == Din == D)
    # b: torch.Tensor,  # (NGI, NH, D) biases
    states_all: torch.Tensor,  # (T+1, NS, B, NH, D) all states
    gates_all: torch.Tensor,  # (T, NGI, B, NH, D) all gates
    backward_pointwise: Callable,
    backward_recurrent_clip_val: float | None = None,
) -> tuple[
    torch.Tensor,  # (NS, B, NH, D) delta errors to initial states
    torch.Tensor,  # (B, T, NGI, NH, D) delta errors to inputs
    torch.Tensor,  # (NGR, NH, Dout, Din) delta errors to recurrent weights
    torch.Tensor,  # (NH, NGI, D) delta errors to biases
]:
    # support the case where states initial has the time dimension explicitly
    T_dim_explicit = False
    if delta_states_last_outside.ndim == 5:
        assert delta_states_last_outside.shape[0] == 1, f"states_initial.shape[0] must be 1: got {delta_states_last_outside.shape}."
        T_dim_explicit = True
        delta_states_last_outside = delta_states_last_outside[0]

    T, NS, B, NH, D = delta_states_all_outside.shape
    NGR, _, _, _ = R.shape
    dtype = R.dtype
    device = R.device

    assert R.shape[1:] == (NH, D, D)
    # assert b.shape == (NGR, NH, D)

    delta_states_tplus1 = delta_states_last_outside  # (NS, B, NH, D)

    R = rearrange(R, "ngr nh dout din -> nh (ngr dout) din")

    delta_R = torch.zeros([NH, NGR * D, D], dtype=dtype, device=device)
    delta_Wx = torch.zeros([T, NGR, B, NH, D], dtype=dtype, device=device)

    for t in range(T - 1, -1, -1):
        delta_states_t_out = delta_states_all_outside[t]  # (NS, B, NH, D)

        states_t = states_all[t + 1]  # (NS, B, NH, D)
        states_tminus1 = states_all[t]  # (NS, B, NH, D)
        gates_t = gates_all[t]  # (NGI, B, NH, D)

        delta_states_tminus1, delta_gates_t = backward_pointwise(
            delta_states_tplus1=delta_states_tplus1,
            delta_states_t_out=delta_states_t_out,
            states_t=states_t,
            states_tminus1=states_tminus1,
            gates_t=gates_t,
        )

        delta_h_tminus1 = (
            rearrange(delta_gates_t, "ngr b nh dout -> nh b (ngr dout)") @ R
        )  # (NH B DHin)
        delta_states_tminus1[0, :, :, :] = rearrange(
            delta_h_tminus1, "nh b din -> b nh din"
        )

        h_tminus1 = states_tminus1[0]  # (B, NH, D)
        delta_R += rearrange(
            delta_gates_t, "ngr b nh dout -> nh (ngr dout) b"
        ) @ rearrange(h_tminus1, "b nh din -> nh b din")

        delta_Wx[t, :, :, :, :] = delta_gates_t

        delta_states_tplus1 = delta_states_tminus1

    delta_Wx = rearrange(delta_Wx, "t ngr b nh d -> b t ngr nh d")

    delta_b = delta_Wx.sum(0).sum(0)  # sum over batch and time dimension # (NGR, NH, D)

    delta_R = rearrange(delta_R, "nh (ngr dout) din -> ngr nh dout din", ngr=NGR)

    if T_dim_explicit:
        delta_states_tminus1 = delta_states_tminus1.unsqueeze(0)

    return delta_states_tminus1, delta_Wx, delta_R, delta_b


def lstm_pointwise_bw(
    delta_states_tplus1,  # (NS=2, B, NH, D)
    delta_states_t_out,  # (NS=2, B, NH, D)
    states_t: torch.Tensor,  # (NS=2, B, NH, D) states (h, c)
    states_tminus1: torch.Tensor,  # (NS=2, B, NH, D) states (h, c)
    gates_t: torch.Tensor,  # (NG=4, B, NH, D) gates (activated) (i, f, z, o)
) -> tuple[
    torch.Tensor,  # (NS=2, B, NH, D) delta states_tminus1 (h, c) (h=0 at this point)
    torch.Tensor,  # (NG=4, B, NH, D) delta gates_t (i, f, z, o)
]:
    delta_states_t = delta_states_t_out + delta_states_tplus1  # (NS=2, B, NH, D)

    i, f, z, o = gates_t.unbind(dim=0)
    h_t, c_t = states_t.unbind(dim=0)
    h_tminus1, c_tminus1 = states_tminus1.unbind(dim=0)

    delta_h_t, delta_c_t = delta_states_t.unbind(dim=0)  # (B, NH, D)
    delta_c_t = delta_c_t + delta_h_t * o * (1 - torch.square(torch.tanh(c_t)))

    delta_o = delta_h_t * torch.tanh(c_t) * (1 - o) * o
    delta_f = delta_c_t * c_tminus1 * (1 - f) * f
    delta_i = delta_c_t * z * (1 - i) * i
    delta_z = delta_c_t * i * (1 - torch.square(z))

    delta_c_tminus1 = delta_c_t * f

    delta_states_tminus1 = torch.stack(
        (torch.zeros_like(delta_c_tminus1), delta_c_tminus1), dim=0
    )
    delta_gates = torch.stack(
        (delta_i, delta_f, delta_z, delta_o), dim=0
    )  # (NG, B, NH, D)

    return delta_states_tminus1, delta_gates


def slstm_pointwise_bw(
    delta_states_tplus1,  # (NS=4, B, NH, D)
    delta_states_t_out,  # (NS=4, B, NH, D)
    states_t: torch.Tensor,  # (NS=4, B, NH, D) states (h, c, n, m)
    states_tminus1: torch.Tensor,  # (NS=4, B, NH, D) states (h, c, n, m)
    gates_t: torch.Tensor,  # (NG=4, B, NH, D) gates (activated) (i, f, z, o)
) -> tuple[
    torch.Tensor,  # (4, B, NH, D) new states (h, c, n, m)
    torch.Tensor,  # (4, B, NH, D) gates (i, f, z, o)
]:
    delta_states_t = delta_states_t_out + delta_states_tplus1  # (NS=4, B, NH, D)

    ibar, fbar, z, o = gates_t.unbind(dim=0)

    h_t, c_t, n_t, m_t = states_t.unbind(dim=0)
    h_tminus1, c_tminus1, n_tminus1, m_tminus1 = states_tminus1.unbind(dim=0)

    delta_h_t, delta_c_t, delta_n_t, delta_m_t = delta_states_t.unbind(
        dim=0
    )  # (B, NH, D)

    delta_c_t = delta_c_t + delta_h_t * (o / n_t)
    delta_n_t = delta_n_t - delta_h_t * (o / torch.square(n_t)) * c_t # maybe use h_t / n_t instead

    logfplusm = m_tminus1 + torch.nn.functional.logsigmoid(fbar)

    f = torch.exp(logfplusm - m_t)
    i = torch.exp(ibar - m_t)

    delta_c_tminus1 = delta_c_t * f
    delta_n_tminus1 = delta_n_t * f

    delta_o = delta_h_t * (c_t / n_t) * o * (1 - o)
    delta_f = (
        (delta_c_t * c_tminus1 + delta_n_t * n_tminus1) * f * (1 - torch.sigmoid(fbar))
    )
    delta_i = (delta_c_t * z + delta_n_t) * i
    delta_z = delta_c_t * i * (1 - torch.square(z))

    delta_states_tminus1 = torch.stack(
        (
            torch.zeros_like(delta_c_tminus1), # delta_h_tminus one is calculated outside
            delta_c_tminus1,
            delta_n_tminus1,
            torch.zeros_like(delta_c_tminus1), # delta_m_tminus1 = zero
        ),
        dim=0,
    ) # (NS=4, B, NH, D)

    delta_gates = torch.stack(
        (delta_i, delta_f, delta_z, delta_o), dim=0 
    ) # (NG=4, B, NH, D)

    return delta_states_tminus1, delta_gates
