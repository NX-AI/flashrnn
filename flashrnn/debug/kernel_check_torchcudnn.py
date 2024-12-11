import os
import torch
from tqdm import tqdm

import sys

print("CUDNN Enabled:", torch.backends.cudnn.enabled)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


device = "cuda"
dtype = torch.float16
B = 16  # batch size
T = 1024  # sequence length
NG = 4  # number of gates (NGI == NGR)
NH = 1  # number of heads
D = 768  # input/hidden (embedding) dimension
NS = 2  # number of states (c, h)

###
WARMUP_ITERS = 1
ITERS = 4

x = torch.randn([T, B, D], device=device, dtype=dtype)

lstm_mod = torch.nn.LSTM(
    D, D, 1, bidirectional=False, device=device, dtype=dtype, batch_first=False
)

for _ in tqdm(range(WARMUP_ITERS), desc="Warmup - CUDnn"):
    out = lstm_mod(x)[0].sum().backward()

for _ in tqdm(range(ITERS), desc="CUDnn"):
    out = lstm_mod(x)[0].sum().backward()
