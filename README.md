# FlashRNN: Optimizing Traditional RNNs on Modern Hardware
Korbinian Pöppel, Maximilian Beck, Sepp Hochreiter

## Intro

FlashRNN implements traditional RNNs like LSTMs, GRUs and Elman networks as well as the recent sLSTM architecture in CUDA and Triton. In contrary to common modern sequence models they have state tracking capabilities (Merrill et al., 2024). All of them are of the basic recurrent structure with input $\mathbf{x}^{(n)}\_t$, bias $\mathbf{b}^{(n)}$, recurrent matrix $\mathbf{R}^{(n)}$ :

$$
\mathbf{g}^{(n)}\_{t} = \mathbf{R}^{(n)} \ \mathbf{s}^{(0)}\_{t-1} + \mathbf{x}^{(n)}_{t} + \mathbf{b}^{(n)} 
$$

$$
\mathbf{y}^{(m)}\_t = \mathcal{P}^{(m)}\left( \left( \mathbf{s}^{(m')}\_{t-1} \right)\_{m' \in \{1..N_s\}} , \left( \mathbf{g}^{(n)}\_{t}  \right)\_{n \in \{1..N_g\}} \right)
$$

Typically the inputs are modified with a linear layer which is omitted here for flexibility (it would look like $\mathbf{x}^{n}\_t = \mathbf{W}^{n} \mathbf{x'}\_t$). This operation can be parallelized along the sequence dimension in contrary to the recurrent part, \\
It employs a multi-head structure, which is equivalent to having a block-diagonal recurrent matrix. The hidden state and gate vectors of hidden dimension $d$ are split into heads of head dimension $d\_{head}$. 

For the fused `triton` backend, heads are limited to small head dimensions $d_{head} \leq 64$. For the CUDA backend there are two versions. The basic `cuda` one that alternates between recurrent matrix multiplication the non-linear pointwise function $\mathcal{P}$ application. This version is not limited in head dimension $d_{head}$. The second is a `cuda_fused` version, which fuses matrix multiplication with point-wise non-linearity into one CUDA kernel using `wmma` instructions and custom caching on SRAM / registers (similar to FlashAttention (Dao et al., 2022), but with a different focus here). Since the recurrent matrix $\mathbf{R}$ and biases $\mathbf{b}$ are used for for every time step, they are customly cached in registers and SRAM, enabling a $2 \times$ to $5 \times$
speedup over the alternating option. 

## Speed comparison

![speed_comparison](head_dim--lstm.svg)

## Installation

To install FlashRNN, simply use:
```bash
pip install flashrnn
``` 

Your hardware needs to support CUDA Compute Capability $8.0$ or later. Make sure, you have an up to date `g++` compiler installed. We recommend to use `conda` with an environment derived from the provided `environment_pt240cu124.yaml`:
```bash
conda env create -n flashrnn -f environment_pt240cu124.yaml
```

To make sure torch uses only compatible compilation flags, you might have to use:
```bash
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"
```

For all kinds of custom setups with torch and CUDA, keep in mind that versions have to match. Also, to make sure the correct CUDA libraries are included you can use the "FLASHRNN_EXTRA_INCLUDE_PATHS" environment variable now to inject different include paths, e.g.:

```bash
export FLASHRNN_EXTRA_INCLUDE_PATHS='/usr/local/include/cuda/:/usr/include/cuda/'
```

or within python:

```python
import os
os.environ['FLASHRNN_EXTRA_INCLUDE_PATHS']='/usr/local/include/cuda/:/usr/include/cuda/'
```



## Using FlashRNN

FlashRNN employs a functional structure, none of the parameters are tied to the `flashrnn` function. To apply it simply use:
```python
import torch
from flashrnn import flashrnn

device = torch.device('cuda')
dtype = torch.bfloat16
B = 8        # batch size
T = 1024     # sequence length
N = 3        # number of heads
D = 256      # head dimension
G = 4        # number of gates / pre-activations for LSTM example
S = 2        # number of states

Wx = torch.randn([B, T, G, N, D], device=device, dtype=dtype, requires_grad=True)
R = torch.randn([G, N, D, D], device=device, dtype=dtype, requires_grad=True)
b = torch.randn([G, N, D], device=device, dtype=dtype, requires_grad=True)
states_initial = torch.randn([S, B, 1, N, D], device=device, dtype=dtype, requires_grad=True)

# available functions
# lstm, gru, elman, slstm

# available backend
# cuda_fused, cuda, triton and vanilla

states, last_states = flashrnn(Wx, R, b, states=states_initial, function="lstm", backend="cuda_fused")

# for LSTM the hidden h state is the first of [h, c]
# [S, B, T, N, D]
hidden_state = states[0]

```
## Acknowledgement 
We thank Thomas Schmied and Pieter-Jan Hoedt for valuable feedback.

## Cite as
```
@misc{pöppel2024flashrnnoptimizingtraditionalrnns,
      title={FlashRNN: I/O-Aware Optimization of Traditional RNNs on modern hardware}, 
      author={Korbinian Pöppel and Maximilian Beck and Sepp Hochreiter},
      year={2024},
      eprint={2412.07752},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.07752}, 
}
```

## License
NXAI Community License (see `LICENSE` file)

## Citations
-  Merrill, W., Petty, J., & Sabharwal, A. (2024). The illusion of state in state-space models. In Proceedings of the Forty-first International Conference on Machine Learning. Retrieved from https://openreview.net/forum?id=QZgo9JZpLq

- Dao, T., Fu, D., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, & A. Oh (Eds.), Advances in Neural Information Processing Systems (Vol. 35, pp. 16344–16359). Curran Associates, Inc. Retrieved from https://proceedings.neurips.cc/paper_files/paper/2022/file/67d57c32e20fd0a7a302cb81d36e40d5-Paper-Conference.pdf


