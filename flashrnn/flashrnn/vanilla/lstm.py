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
    _, c = torch.unbind(states, dim=1)
    iraw, fraw, zraw, oraw = torch.unbind(raw, dim=1)
    ogate = torch.sigmoid(oraw)
    igate = torch.sigmoid(iraw)
    fgate = torch.sigmoid(fraw)
    zval = torch.tanh(zraw)
    cnew = fgate * c + igate * zval
    ynew = ogate * torch.tanh(cnew)

    # shapes ([B,H], [B,H], [B,H]), ([B,H],[B,H],[B,H],[B,H])
    return torch.stack((ynew, cnew), dim=1), torch.stack(
        (igate, fgate, zraw, ogate), dim=1
    )
