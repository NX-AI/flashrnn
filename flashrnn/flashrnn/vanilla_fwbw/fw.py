# Maximilian Beck
from typing import Callable

import torch
from einops import rearrange

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


def forward_sequence(
    states_initial: torch.Tensor,  # (NS, B, NH, D) initial states
    Wx: torch.Tensor,  # (B, T, NGI, NH, D) inputs
    R: torch.Tensor,  # (NGR, NH, Dout, Din) recurrent weights (Dout == Din == D)
    b: torch.Tensor,  # (NH, NGI, D) biases
    forward_pointwise: Callable[
        [
            torch.Tensor,  # (B, NGI, NH, D) Wx
            torch.Tensor,  # (B, NGR, NH, D) Rh
            torch.Tensor,  # (NGI, NH, D) b
            torch.Tensor,  # (B, NS, NH, D) states
        ],
        tuple[
            torch.Tensor,  # (NS, B, NH, D) new states
            torch.Tensor,  # (NGI, B, NH, D) gates
        ],
    ],
    output_gates_and_states_initial: bool = False,  # If true, return all states including the initial state and the gates (used for backward)
) -> (  # either return only states + last state
    tuple[
        torch.Tensor,  # (T, NS, B, NH, D) all states
        torch.Tensor,  # (NS, B, NH, D) last state
    ]  # or return states + last state + gates
    | tuple[
        tuple[
            torch.Tensor,  # (T+1, NS, B, NH, D) all states (T+1 since the initial states are included)
            torch.Tensor,  # (NS, B, NH, D) last state
        ],
        torch.Tensor,  # (T, NGI, B, NH, D) gates
    ]
):
    # support the case where states initial has the time dimension explicitly
    T_dim_explicit = False
    if states_initial.ndim == 5:
        T_dim_explicit = True
        states_initial = states_initial[0]
    NS, B, NH, D = states_initial.shape
    _, T, NGI, _, _ = Wx.shape
    NGR, _, _, _ = R.shape
    assert R.shape[1:] == (NH, D, D)
    assert b.shape == (NGI, NH, D), f"{b.shape} != (NGI, NH, D)={(NGI, NH, D)}"

    states_all = torch.zeros([T + 1, NS, B, NH, D], device=Wx.device, dtype=Wx.dtype)

    if output_gates_and_states_initial:
        gates_all = torch.zeros([T, NGI, B, NH, D], device=Wx.device, dtype=Wx.dtype)

    states_all[0] = states_initial
    states = states_initial

    R = rearrange(R, "ngr nh dout din -> 1 nh din (ngr dout)")

    for t in range(T):
        Rh = states[0].reshape(B, NH, 1, D) @ R
        Rh = Rh.reshape(B, NH, NGR, D).transpose(1, 2)  # (B, NGR, NH, D)
        states, gates = forward_pointwise(Wx[:, t, ...], Rh, b, states)
        states_all[t + 1] = states
        if output_gates_and_states_initial:
            gates_all[t] = gates

    if T_dim_explicit:
        states_last = states_all[-1:]
    else:
        states_last = states_all[-1]
    if output_gates_and_states_initial:
        return (states_all, states_last), gates_all
    return states_all[1:], states_last


def lstm_pointwise_fw(
    Wx: torch.Tensor,  # (B, 4, NH, D) inputs (i, f, z, o)
    Rh: torch.Tensor,  # (B, 4, NH, D) hidden states times recurrent weights (i, f, z, o)
    b: torch.Tensor,  # (4, NH, D) biases
    states: torch.Tensor,  # (B, 2, NH, D) states (h, c)
) -> tuple[
    torch.Tensor,  # (2, B, NH, D) new states (h, c)
    torch.Tensor,  # (4, B, NH, D) gates (i, f, z, o)
]:
    gatesbar = Wx + Rh + b[None, :]

    ibar, fbar, zbar, obar = gatesbar.unbind(dim=1)

    h, c = states.unbind(dim=0)

    i = torch.sigmoid(ibar)
    f = torch.sigmoid(fbar)
    z = torch.tanh(zbar)
    o = torch.sigmoid(obar)

    c_next = f * c + i * z
    h_next = o * torch.tanh(c_next)

    states_next = torch.stack((h_next, c_next), dim=0)
    gates_next = torch.stack((i, f, z, o), dim=0)

    return states_next, gates_next


def slstm_pointwise_fw(
    Wx: torch.Tensor,  # (B, 4, NH, D) inputs (i, f, z, o)
    Rh: torch.Tensor,  # (B, 4, NH, D) hidden states times recurrent weights (i, f, z, o)
    b: torch.Tensor,  # (4, NH, D) biases
    states: torch.Tensor,  # (B, 4, NH, D) states (h, c, n, m)
) -> tuple[
    torch.Tensor,  # (4, B, NH, D) new states (h, c, n, m)
    torch.Tensor,  # (4, B, NH, D) gates (i, f, z, o)
]:
    gatesbar = Wx + Rh + b[None, :]

    ibar, fbar, zbar, obar = gatesbar.unbind(dim=1)

    h, c, n, m = states.unbind(dim=0)
    # with torch.no_grad():
    logfplusm = m + torch.nn.functional.logsigmoid(fbar)
    if torch.all(n == 0.0):
        m_next = ibar
    else:
        m_next = torch.maximum(logfplusm, ibar)

    f = torch.exp(logfplusm - m_next)
    i = torch.exp(ibar - m_next)
    z = torch.tanh(zbar)
    o = torch.sigmoid(obar)

    c_next = f * c + i * z
    n_next = torch.maximum(f * n + i, torch.ones_like(n))

    h_next = o * (c_next / n_next)

    states_next = torch.stack((h_next, c_next, n_next, m_next), dim=0)
    gates_next = torch.stack(
        (ibar, fbar, z, o), dim=0
    )  # Note: we store ibar and fbar for backward

    return states_next, gates_next
