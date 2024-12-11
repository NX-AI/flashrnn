import torch
import triton.language as tl

_torch_to_triton_dtype = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}

def torch2triton_dtype(dtype):
    return _torch_to_triton_dtype[dtype]

def is_power_of_2(n):
    assert isinstance(n, int)
    return (n & (n - 1)) == 0

def next_multiple_of(n, multiple_of):
    assert isinstance(n, int)
    assert isinstance(multiple_of, int)
    return ((n + multiple_of - 1) // multiple_of) * multiple_of