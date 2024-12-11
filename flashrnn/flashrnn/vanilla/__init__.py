# Copyright 2024 NXAI GmbH
# Korbinian Poeppel
from typing import Callable
import torch

from .slstm import flashrnn_forward_pointwise as flashrnn_forward_pointwise_slstm
from .lstm import flashrnn_forward_pointwise as flashrnn_forward_pointwise_lstm
from .elman import flashrnn_forward_pointwise as flashrnn_forward_pointwise_elman
from .gru import flashrnn_forward_pointwise as flashrnn_forward_pointwise_gru


flashrnn_pointwise_function_registry: dict[str, Callable] = {
    "slstm": flashrnn_forward_pointwise_slstm,
    "lstm": flashrnn_forward_pointwise_lstm,
    "elman": flashrnn_forward_pointwise_elman,
    "gru": flashrnn_forward_pointwise_gru,
}


def flashrnn_forward(
    Wx: torch.Tensor,  # [T, B, GI, H, I]
    states: torch.Tensor,  # [T, B, ST, H, D] only the first is used for recurrence! T is one (indicates the time dimension)
    R: torch.Tensor,  # [H, P, GR, D] - H num_heads, GR gates, D new hidden for gates, , P previous hidden (=D)
    b: torch.Tensor,  # [H, GT, D]
    pointwise_forward: Callable[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]],
        tuple[torch.Tensor, torch.Tensor],
    ],
    constants: dict[str, float] = {},
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_states = states.shape[2]
    sequence_dim = Wx.shape[0]
    num_gates_r = R.shape[2]
    batch_dim = Wx.shape[1]
    num_heads = R.shape[0]
    head_dim = R.shape[3]

    assert batch_dim == states.shape[1]

    # g = torch.zeros(
    #     [sequence_dim + 1, batch_dim, num_gates_t, num_heads, head_dim],
    #     device=Wx.device,
    #     dtype=Wx.dtype,
    # )

    states_all = torch.zeros(
        [sequence_dim + 1, batch_dim, num_states, num_heads, head_dim],
        device=Wx.device,
        dtype=Wx.dtype,
    )
    states = states[0]
    states_all[0] = states
    R = R.reshape(1, num_heads, head_dim, num_gates_r * head_dim)
    for i, Wx_t in enumerate(Wx.unbind(dim=0)):
        Ry = (
            states[:, 0]
            .reshape(batch_dim, num_heads, 1, -1)
            .matmul(R)
            .reshape(batch_dim, num_heads, num_gates_r, head_dim)
            .transpose(1, 2)
        )

        states, _ = pointwise_forward(Wx_t, Ry, b, states, constants=constants)
        # g[i] = gates
        states_all[i + 1] = states

    # shapes ([T, B, S, N, H], [S, B, 4, N, H])
    return states_all[1:].view(
        sequence_dim, batch_dim, num_states, num_heads, head_dim
    ), states.view(1, batch_dim, num_states, num_heads, head_dim)


def flashrnn_forward_step(
    Wx: torch.Tensor,  # [B, GI, I]
    states: torch.Tensor,  # [ST, B, D] only the first is used for recurrence!
    R: torch.Tensor,  # [H, GR, D, D] - K num_heads
    b: torch.Tensor,  # [H, GT, D]
    pointwise_forward: Callable[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]],
        tuple[torch.Tensor, torch.Tensor],
    ],
    constants: dict[str, float] = {},
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_states = states.shape[0]
    sequence_dim = Wx.shape[0]
    num_gates_r = R.shape[1]
    hidden_dim = R.shape[2] * R.shape[0]
    batch_dim = Wx.shape[1]
    num_heads = R.shape[0]
    head_dim = R.shape[2]

    assert batch_dim == states.shape[1]
    assert hidden_dim == states.shape[2]

    # g = Wx.zeros(
    #     [sequence_dim + 1, num_gates_t, batch_dim, hidden_dim],
    # )
    R = (
        R.reshape(num_heads, num_gates_r * head_dim, head_dim)
        .transpose(1, 2)
        .reshape(1, num_heads, head_dim, num_gates_r * head_dim)
    )

    states_all = Wx.zeros(
        [num_states, sequence_dim + 1, batch_dim, hidden_dim],
    )
    states_all[:, 0] = states
    Ry = (
        states[0]
        .reshape(batch_dim, num_heads, 1, -1)
        .matmul(R)
        .reshape(batch_dim, num_heads, num_gates_r, -1)
        .transpose(1, 2)
        .reshape(batch_dim, num_gates_r, -1)
    )
    sdtype = states.dtype
    states, gates = pointwise_forward(Wx[0], Ry, b, states, constants=constants)
    states = states.to(dtype=sdtype)

    # shapes ([S, B, H], ([B,H], [B,H], [B,H]), [S, B, 4*H])
    return states.view(1, batch_dim, num_states, num_heads, head_dim)
