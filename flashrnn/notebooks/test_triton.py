import sys

sys.path.append("../..")
import torch
from flashrnn.tests.utils import model_test


if __name__ == "__main__":
    device = "cuda"
    dtype = torch.float32
    TGT_DTYPE = torch.float32
    B = 3  # batch size
    T = 23  # sequence length
    NG = 4  # number of gates (NGI == NGR)
    NH = 5  # number of heads
    D = 32  # input/hidden (embedding) dimension
    NS = 2  # number of states (c, h)

    model_test(
        batch_size=B,
        sequence_size=T,
        num_heads=NH,
        head_dim=D,
        backend="triton_fused",
        backend_cmp="vanilla",
        function="slstm",
        dtype=dtype,
        include_backward=True,
        tensor_compare_kwargs={"atol": 0.1, "rtol": 0.2},
    )
