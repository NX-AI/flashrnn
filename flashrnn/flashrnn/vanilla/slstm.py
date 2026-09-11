# SPDX-License-Identifier: Apache-2.0
import torch

def flashrnn_forward_pointwise(
    Wx: torch.Tensor,  # dim [B, 4*H]
    Ry: torch.Tensor,  # dim [B, 4*H]
    b: torch.Tensor,  # dim [1, 4*H]
    states: torch.Tensor,  # dim [B, 4, H]
    constants: dict[str, float],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    _ = constants
    raw = Wx + Ry + b
    y, c, n, m = torch.unbind(states, dim=1)
    iraw, fraw, zraw, oraw = torch.unbind(raw, dim=1)
    logfplusm = torch.nn.functional.logsigmoid(fraw) + m
    mnew = torch.where(n==0, iraw, torch.max(iraw, logfplusm))
    ogate = torch.sigmoid(oraw)
    igate = torch.exp(iraw - mnew)
    fgate = torch.exp(logfplusm - mnew)
    cnew = fgate * c + igate * torch.tanh(zraw)
    nnew = torch.maximum(fgate * n + igate, torch.ones_like(n))
    ynew = ogate * cnew / nnew

    # shapes ([B,H], [B,H], [B,H]), ([B,H],[B,H],[B,H],[B,H])
    return torch.stack((ynew, cnew, nnew, mnew), dim=1), torch.stack(
        (igate, fgate, zraw, ogate), dim=1
    )
