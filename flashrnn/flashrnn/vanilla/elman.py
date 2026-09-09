# SPDX-License-Identifier: Apache-2.0
import torch


def flashrnn_forward_pointwise(
    Wx: torch.Tensor,  # dim [B, 4, H, D]
    Ry: torch.Tensor,  # dim [B, 4, H, D]
    b: torch.Tensor,  # dim [4, H, D]
    states: torch.Tensor,  # dim [B, 2, H, D]
    constants: dict[str, float],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    _ = constants
    raw = Wx + Ry + b[None, :]
    (graw,) = torch.unbind(raw, dim=1)
    ynew = torch.tanh(graw)

    # shapes ([B,H], [B,H], [B,H]), ([B,H],[B,H],[B,H],[B,H])
    return torch.stack((ynew,), dim=1), torch.stack((graw,), dim=1)
