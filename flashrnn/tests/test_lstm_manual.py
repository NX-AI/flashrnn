import torch
from flashrnn.flashrnn.flashrnn import flashrnn, _get_config


if __name__ == "__main__":
    ending = ""

    Wx = torch.load(f"tensor_Wx{ending}.pt", weights_only=True, map_location="cuda:0")
    b = torch.load(f"tensor_b{ending}.pt", weights_only=True, map_location="cuda:0").to(
        dtype=torch.bfloat16
    )
    R = torch.load(f"tensor_R{ending}.pt", weights_only=True, map_location="cuda:0").to(
        dtype=torch.bfloat16
    )
    h = torch.load(f"tensor_h{ending}.pt", weights_only=True, map_location="cuda:0")
    states = torch.load(
        f"tensor_s{ending}.pt", weights_only=True, map_location="cuda:0"
    )
    cfg = _get_config(Wx, R, b, "lstm", backend="cuda_fused", dtype="bfloat16")
    h2 = flashrnn(Wx, R, b, states, function="lstm", config=cfg)

    print(Wx.dtype, b.dtype, R.dtype, h.dtype, states.dtype)
    print(Wx.shape, R.shape, b.shape)
    print(h.shape, h2[0].shape)
    print(torch.any(torch.isnan(h)))
    print(torch.any(torch.isnan(h2[0])))
    print(h - h2[0])
