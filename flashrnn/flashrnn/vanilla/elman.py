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
    # B, H, D = Wx.shape[0], Wx.shape[2], Wx.shape[3]
    raw = Wx + Ry + b[None, :]
    # raw = raw.view(-1, 4, -1)
    (graw,) = torch.unbind(raw, dim=1)
    # with torch.no_grad():  # THE difference to maxg aka max_gradient (here max / max_static)
    ynew = torch.tanh(graw)

    # shapes ([B,H], [B,H], [B,H]), ([B,H],[B,H],[B,H],[B,H])
    return torch.stack((ynew,), dim=1), torch.stack((graw,), dim=1)
