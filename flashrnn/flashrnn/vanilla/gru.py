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
    # gates g for recurrent only
    graw = Ry[:, 0] + b[None, 0]
    # gates r, z for for input and recurrent
    rraw, zraw = Wx[:, 0] + Ry[:, 1] + b[None, 1], Wx[:, 1] + Ry[:, 2] + b[None, 2]
    # gates n for input only
    nraw = Wx[:, 2] + b[None, 3]
    (h,) = torch.unbind(states, dim=1)
    # raw = raw.view(-1, 4, -1)

    ggate = graw
    ngate = nraw
    rgate = torch.sigmoid(rraw)
    zgate = torch.sigmoid(zraw)

    ynew = zgate * h + (1 - zgate) * torch.tanh(ngate + rgate * ggate)

    # shapes ([B,H], [B,H], [B,H]), ([B,H],[B,H],[B,H],[B,H])
    return torch.stack((ynew,), dim=1), torch.stack((graw, rraw, zraw, nraw), dim=1)
