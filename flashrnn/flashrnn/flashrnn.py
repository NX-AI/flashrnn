# Copyright 2024 NXAI GmbH
# Korbinian Poeppel
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence, Union

import torch
from torch.autograd.function import once_differentiable

from ..autotune.constrint import ValueHeuristic, ValueRefinement
from .cuda_init import load
from .cuda_init_parametric import load_parametric_and_test_and_bisect
from .gpu_info.gpu_info import get_gpu_info
from .vanilla import (
    flashrnn_forward,
    flashrnn_forward_step,
    flashrnn_pointwise_function_registry,
)

LOGGER = logging.getLogger(__name__)


DTYPE_DICT = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}
DTYPE_DICT_REV = {
    torch.bfloat16: "bfloat16",
    torch.float: "float32",
    torch.float16: "float16",
    torch.float64: "float64",
}

DTYPES = Literal["bfloat16", "float16", "float32"]

curdir = Path(os.path.split(os.path.os.path.abspath(__file__))[0])

kernel_registry = {}

# maps the rnn function to the following values
#   gates_states = (num_gates_r, num_gates_w, num_gates_i, num_gates_t, num_states)
#   constants = {}, # whatever the function needs
#   simple_agg = True/False # whether the function simply adds pre-activations Wx + Ry
#   initial_val = 0.  # initial value of the states
rnn_function_registry = {
    # standard variants, all connect
    "slstm": {
        "gates_states": (4, 4, 4, 4, 4),
        "constants": {},
        "simple_agg": True,
    },
    "lstm": {
        "gates_states": (4, 4, 4, 4, 2),
        "constants": {},  # no constants here
        "simple_agg": True,
    },
    "gru": {
        "gates_states": (3, 3, 4, 4, 1),
        "constants": {},  # no constants here
        "simple_agg": False,
    },
    "elman": {
        "gates_states": (1, 1, 1, 1, 1),
        "constants": {},  # no constants here
        "simple_agg": True,
    },
}

_python_dtype_to_cuda_dtype = {
    "float32": "float",
    "float": "float",
    "float16": "__half",
    "bfloat16": "__nv_bfloat16",
}


def round_to_multiple(n, m=8):
    return ((n + m - 1) // m) * m


def conditional_decorator(condition, decorator):
    """A higher-order decorator that applies 'decorator' only if 'condition' is True."""

    def dummy_decorator(func):
        """A dummy decorator that does nothing."""
        return func

    if condition:
        # If condition is True, return the actual decorator
        return decorator
    else:
        # If condition is False, return the dummy decorator
        return dummy_decorator


def permute_to(input_shape, output_shape) -> Optional[list[int]]:
    """
    >>> permute_to("ABC", "BAC")
    (1, 0, 2)
    """
    if input_shape == output_shape:
        return None
    p = []
    for x in output_shape:
        p.append(input_shape.index(x))
    return tuple(p)


def round_to_divisible(x, y):
    """
    Round a number such that round(x) divides y
    """
    xnew = x
    while y % xnew != 0:
        znew = y // xnew
        xnew = (y + znew - 1) // znew
    return xnew


@dataclass
class FlashRNNConfig:
    backend: Literal[
        "vanilla", "vanilla_fwbw", "cuda", "cuda_fused", "triton_fused"
    ] = "cuda_fused"
    # the type of function a cell computes
    function: str = "lstm"
    # this option cuts of the gradient for recurrent connection, i.e. no exploding gradient if False
    gradient_recurrent_cut: bool = False
    # this option clips the gradient values for recurrent connections at dy
    gradient_recurrent_clipval: Optional[float] = None
    # this option clips the y value
    forward_clipval: Optional[float] = None
    # additional scalar constants that might be modified
    constants: Optional[dict[str, float]] = None
    # whether all gate aggregations are of the type "R @ y + W @ x + b" or if there is a function g_r(R @ y) involved
    # this roughly doubles the memory needed to be stored for backward
    simple_agg: bool = True

    hidden_dim: int = -1
    num_heads: int = -1  # this must divide the hidden size, is not yet supported by all versions in this directory
    head_dim: int = -1  # alternative to num_heads, equals to hidden_dim // num_heads
    num_states: int = 4  # this is for the sLSTM, a standard LSTM  has 2

    num_gates_r: int = 4  # how many gates take recurrent input
    num_gates_w: int = 4  # how many gates take external input
    num_gates_i: int = (
        4  # how many gates interact between cells (i.e. r and w together)
    )
    num_gates_t: int = 4  # how many gates are there in total (including biases only)
    # the gate order is as follows in case some are reduced (i.e. gates_r)
    # [gates_r ... ...]
    # [... gates_w ...]
    # [  gates_i   ...]
    # [    gates_t    ]

    # this can be ignored internally, but may be used to optimize kernels
    batch_size: int = 8

    # B = batch, T time/sequence dim, N num heads, S state dimension, P previous D dimension (R matrix)
    # D head dim or hidden dim, G gates
    input_shape: Literal["BTGHD", "TBGHD"] = "BTGHD"
    output_shape: Literal[
        "SBHTD",
        "STBHD",
        "STBHD",
    ] = "SBTHD"

    recurrent_shape: Literal["GHDP", "HGDP", "HPGD", "HPDG"] = "GHDP"
    bias_shape: Literal["GHD", "HGD", "HDG"] = "GHD"

    # internal shapes are overwritten by backend
    # if you use the shape when calling from outside you minimize transposes
    _internal_input_shape: Literal["TBGHD", "TBHGD", "TBHDG"] = "TBGHD"
    _internal_recurrent_shape: Literal["GHDP", "HGDP", "HPGD", "HPDG"] = "GHDP"
    _internal_bias_shape: Literal["GHD, HGD, HDG"] = "HDG"
    _internal_output_shape: Literal["TSHBD", "BTSHD"] = "STBHD"

    _internal_input_permutation: Optional[tuple[int, int, int, int, int]] = None
    _internal_recurrent_permutation: Optional[tuple[int, int, int, int]] = None
    _internal_bias_permutation: Optional[tuple[int, int, int]] = None
    _internal_output_permutation: Optional[tuple[int, int, int, int, int]] = None
    _internal_output_backward_permutation: Optional[tuple[int, int, int, int]] = None

    # this is moved to slstm
    # backend: str = "vanilla"
    dtype: DTYPES = "bfloat16"
    dtype_acc: Optional[DTYPES] = "float32"
    dtype_b: Optional[DTYPES] = None  # biases
    dtype_r: Optional[DTYPES] = None  # recurrent matrix
    dtype_w: Optional[DTYPES] = None  # inputs / w matrix
    dtype_g: Optional[DTYPES] = None  # gates
    dtype_s: Optional[DTYPES] = None  # states
    dtype_a: Optional[DTYPES] = None  # internal accumulation
    # if this is set to true, the kernel dtype has to match all other dtypes
    # but input dtypes might be arbitrary (are autocasted)
    enable_automatic_mixed_precision: bool = True
    trainable_r: Union[list[bool], bool] = True
    trainable_b: Union[list[bool], bool] = True
    # initial value for each state
    initial_val: Union[float, Sequence[float]] = 0.0

    _internal_dtype_size: int = 2
    _internal_acc_dtype_size: int = 4

    @property
    def input_dim(self):
        return self.num_gates_w * self.hidden_dim

    @property
    def torch_dtype(self) -> torch.dtype:
        return DTYPE_DICT[self.dtype]

    @property
    def torch_dtype_b(self) -> torch.dtype:
        return DTYPE_DICT[self.dtype_b]

    @property
    def torch_dtype_r(self) -> torch.dtype:
        return DTYPE_DICT[self.dtype_r]

    @property
    def torch_dtype_w(self) -> torch.dtype:
        return DTYPE_DICT[self.dtype_w]

    @property
    def torch_dtype_s(self) -> torch.dtype:
        return DTYPE_DICT[self.dtype_s]

    def __post_init__(self):
        if self.dtype_acc != "float32":
            assert self.dtype_acc == "float16" and self.dtype == "float16"

        if self.dtype_acc == "float32":
            self._internal_acc_dtype_size = 4
        else:
            self._internal_acc_dtype_size = 2
        if self.dtype == "float32":
            self._internal_dtype_size = 4
        else:
            self._internal_dtype_size = 2

        if self.num_heads <= 0 and self.head_dim <= 0:
            self.num_heads = 1
        if self.num_heads <= 0:
            self.num_heads = self.hidden_dim // self.head_dim
        elif self.head_dim <= 0:
            self.head_dim = self.hidden_dim // self.num_heads
        if self.hidden_dim <= 0:
            self.hidden_dim = self.num_heads * self.head_dim
        if self.num_gates_t < 0:
            self.num_gates_t = self.num_gates_r
        if self.dtype_b is None:
            self.dtype_b = self.dtype
        if self.dtype_a is None:
            self.dtype_a = self.dtype_b
        if self.dtype_r is None:
            self.dtype_r = self.dtype
        if self.dtype_w is None:
            self.dtype_w = self.dtype
        if self.dtype_s is None:
            self.dtype_s = self.dtype_w
        if self.dtype_g is None:
            self.dtype_g = self.dtype_r

        assert (
            self.function in rnn_function_registry
        ), f"RNN function {self.function} not in registry"
        (
            self.num_gates_r,
            self.num_gates_w,
            self.num_gates_i,
            self.num_gates_t,
            self.num_states,
        ) = rnn_function_registry[self.function]["gates_states"]
        # TODO fix this by padding the recurrent matrix
        if self.function == "gru" and self.backend == "cuda_fused":
            LOGGER.info(
                "Fixing cuda_fused to cuda kernel as recurrent gates do not divide 8 which is problematic for kernels"
            )
            self.backend = "cuda"
        if self.constants is None:
            self.constants = rnn_function_registry[self.function]["constants"]
        self.simple_agg = rnn_function_registry[self.function]["simple_agg"]
        if "initial_val" in rnn_function_registry[self.function]:
            self.initial_val = rnn_function_registry[self.function]["initial_val"]

        if self.backend == "vanilla":
            self._internal_input_shape = "TBGHD"
            self._internal_bias_shape = "GHD"
            self._internal_recurrent_shape = "HPGD"
            self._internal_output_shape = "TBSHD"
        elif self.backend == "vanilla_fwbw":
            self._internal_input_shape = "BTGHD"
            self._internal_recurrent_shape = "GHDP"
            self._internal_bias_shape = (
                "GHD"  # TODO should be HGD but permute does not work
            )
            self._internal_output_shape = "TSBHD"
        elif self.backend == "cuda":
            self._internal_input_shape = "TBHGD"
            self._internal_recurrent_shape = "HPGD"
            self._internal_bias_shape = "HGD"
            self._internal_output_shape = "STBHD"
        elif self.backend == "cuda_fused":
            self._internal_input_shape = "TBHDG"
            self._internal_recurrent_shape = "HDGP"
            self._internal_bias_shape = "HDG"
            self._internal_output_shape = "STBHD"
            if self.dtype == "float32":
                self.batch_size = 16
        elif self.backend == "triton_fused":
            self._internal_input_shape = "BTGHD"
            self._internal_recurrent_shape = "GHDP"
            self._internal_bias_shape = (
                "GHD"  # TODO should be HGD but permute does not work
            )
            self._internal_output_shape = "TSBHD"

        self._internal_input_permutation = permute_to(
            self.input_shape, self._internal_input_shape
        )
        self._internal_output_permutation = permute_to(
            self._internal_output_shape, self.output_shape
        )
        self._internal_recurrent_permutation = permute_to(
            self.recurrent_shape, self._internal_recurrent_shape
        )
        self._internal_bias_permutation = permute_to(
            self.bias_shape, self._internal_bias_shape
        )
        self._internal_output_backward_permutation = permute_to(
            self.output_shape, self._internal_output_shape
        )

    @property
    def defines(self):
        return (
            [
                f"-DFLASHRNN_HIDDEN_SIZE={self.hidden_dim}",
                f"-DFLASHRNN_BATCH_SIZE={self.batch_size}",
                f"-DFLASHRNN_NUM_HEADS={self.num_heads}",
                f"-DFLASHRNN_NUM_STATES={self.num_states}",
                f"-DFLASHRNN_DTYPE={_python_dtype_to_cuda_dtype[self.dtype]}",
                f"-DFLASHRNN_DTYPE_B={_python_dtype_to_cuda_dtype[self.dtype_b]}",
                f"-DFLASHRNN_DTYPE_R={_python_dtype_to_cuda_dtype[self.dtype_r]}",
                f"-DFLASHRNN_DTYPE_W={_python_dtype_to_cuda_dtype[self.dtype_w]}",
                f"-DFLASHRNN_DTYPE_G={_python_dtype_to_cuda_dtype[self.dtype_g]}",
                f"-DFLASHRNN_DTYPE_S={_python_dtype_to_cuda_dtype[self.dtype_s]}",
                f"-DFLASHRNN_DTYPE_A={_python_dtype_to_cuda_dtype[self.dtype_a]}",
                f"-DFLASHRNN_NUM_GATES_R={self.num_gates_r}",
                f"-DFLASHRNN_NUM_GATES_W={self.num_gates_w}",
                f"-DFLASHRNN_NUM_GATES_I={self.num_gates_i}",
                f"-DFLASHRNN_NUM_GATES_T={self.num_gates_t}",
                f"-DFLASHRNN_SIMPLE_AGG={'true' if self.simple_agg else 'false'}",
            ]
            + [f"-DFLASHRNN_USE_DTYPE_{self.dtype.upper()}=1"]
            + (
                [
                    "-DFLASHRNN_GRADIENT_RECURRENT_CLIPVAL_VALID=true",
                    f"-DFLASHRNN_GRADIENT_RECURRENT_CLIPVAL={self.gradient_recurrent_clipval}",
                ]
                if self.gradient_recurrent_clipval is not None
                else [
                    "-DFLASHRNN_GRADIENT_RECURRENT_CLIPVAL_VALID=false",
                    "-DFLASHRNN_GRADIENT_RECURRENT_CLIPVAL=0.0",
                ]
            )
            + (
                [
                    "-DFLASHRNN_FORWARD_CLIPVAL_VALID=true",
                    f"-DFLASHRNN_FORWARD_CLIPVAL={self.gradient_recurrent_clipval}",
                ]
                if self.gradient_recurrent_clipval is not None
                else [
                    "-DFLASHRNN_FORWARD_CLIPVAL_VALID=false",
                    "-DFLASHRNN_FORWARD_CLIPVAL=0.0",
                ]
            )
        )


class _FlashRNNCUDAFused:
    mod: dict = {}

    @classmethod
    def instance(cls, config: FlashRNNConfig):
        device_id = torch.cuda.current_device()
        cfgdevstr = repr(config) + f"_device{device_id}"
        if cfgdevstr not in cls.mod:
            gpu_info = get_gpu_info(device_id=torch.cuda.current_device())
            VR = ValueRefinement
            LF = ValueHeuristic.LARGEST_FIRST
            SF = ValueHeuristic.SMALLEST_FIRST
            value_refinements = (
                VR("FLASHRNN_HIDDEN_DIM", LF),
                VR("FLASHRNN_NUM_HEADS", LF),
                VR("FLASHRNN_BACKWARD_SHARED_MEMORY_PADDING", SF),
                VR("FLASHRNN_FORWARD_SHARED_MEMORY_PADDING", SF),
                VR("FLASHRNN_BACKWARD_WARP_TILING_DIM_GATE", LF),
                VR("FLASHRNN_FORWARD_WARP_TILING_DIM_HIDDEN", LF),
                VR("FLASHRNN_BACKWARD_WARP_TILING_DIM_BATCH", LF),
                VR("FLASHRNN_FORWARD_WARP_TILING_DIM_BATCH", LF),
                VR("FLASHRNN_BACKWARD_WARP_TILING_DIM_HIDDEN", LF),
                VR("FLASHRNN_FORWARD_WARP_TILING_DIM_GATE", LF),
                VR("FLASHRNN_BACKWARD_MULTIHEAD_TILING_COUNT", LF),
                VR("FLASHRNN_FORWARD_MULTIHEAD_TILING_COUNT", LF),
                VR("FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_HIDDEN", LF),
                VR("FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE", LF),
                VR("FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_GATE", SF),
                VR("FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN", SF),
                VR("FLASHRNN_BACKWARD_BLOCK_TILING_COUNT_BATCH", LF),
                VR("FLASHRNN_FORWARD_BLOCK_TILING_COUNT_BATCH", LF),
                VR("FLASHRNN_BACKWARD_WARP_LOOPING_COUNT_BATCH", LF),
                VR("FLASHRNN_FORWARD_WARP_LOOPING_COUNT_BATCH", LF),
                VR("FLASHRNN_BACKWARD_WARP_LOOPING_COUNT_HIDDEN", SF),
                VR("FLASHRNN_FORWARD_WARP_LOOPING_COUNT_GATE", SF),
                VR("FLASHRNN_BACKWARD_WARP_TILING_COUNT_GATE", LF),
                VR("FLASHRNN_FORWARD_WARP_TILING_COUNT_HIDDEN", LF),
                VR("FLASHRNN_BACKWARD_WARP_TILING_COUNT_BATCH", LF),
                VR("FLASHRNN_FORWARD_WARP_TILING_COUNT_BATCH", LF),
                VR("FLASHRNN_BACKWARD_WARP_RECURRENT_CACHED_GATE", LF),
                VR("FLASHRNN_FORWARD_WARP_RECURRENT_CACHED_HIDDEN", LF),
            )

            constraint_str = (
                """
                WARP_SIZE == 32;
                """
                + (
                    """
                FLASHRNN_FORWARD_WARP_TILING_DIM_BATCH == [ 8, 16, 32 ];
                FLASHRNN_BACKWARD_WARP_TILING_DIM_BATCH == [ 8, 16, 32 ];
                FLASHRNN_FORWARD_WARP_TILING_DIM_GATE == [ 8, 16, 32 ];
                FLASHRNN_BACKWARD_WARP_TILING_DIM_HIDDEN == [ 8, 16, 32 ];
                FLASHRNN_FORWARD_WARP_TILING_DIM_HIDDEN == 16;
                FLASHRNN_BACKWARD_WARP_TILING_DIM_GATE == 16;
                """
                    if config.dtype == "float16" or config.dtype == "bfloat16"
                    else """
                FLASHRNN_FORWARD_WARP_TILING_DIM_BATCH == [ 16 ];
                FLASHRNN_BACKWARD_WARP_TILING_DIM_BATCH == [ 16 ];
                FLASHRNN_FORWARD_WARP_TILING_DIM_GATE == [ 16 ];
                FLASHRNN_BACKWARD_WARP_TILING_DIM_HIDDEN == [ 16 ];
                FLASHRNN_FORWARD_WARP_TILING_DIM_HIDDEN == 8;
                FLASHRNN_BACKWARD_WARP_TILING_DIM_GATE == 8;
                """
                )
                + f"""
                # Optimize not fully occupied
                MAX_THREADS_PER_BLOCK_FORWARD == {gpu_info['maxThreadsPerBlock'] // 4};
                MAX_THREADS_PER_BLOCK_BACKWARD == {gpu_info['maxThreadsPerBlock'] // 4};
                # Factor 2 as these are float32 registers
                REGISTERS_PER_BLOCK_FORWARD == {gpu_info['regsPerMultiprocessor'] * (2 if config.dtype != 'float32' else 1) } ;
                REGISTERS_PER_BLOCK_BACKWARD == {gpu_info['regsPerMultiprocessor'] * (2 if config.dtype != 'float32' else 1)} ;
                STREAMING_MULTIPROCESSORS == {gpu_info['multiProcessorCount']};
                SHARED_MEMORY_PER_BLOCK == {min(gpu_info['sharedMemPerBlockOptin'] - 1024, 227000)};
                FLASHRNN_NUM_GATES_R == {config.num_gates_r};
                FLASHRNN_NUM_GATES_W == {config.num_gates_w};
                SHARED_MEMORY_PADDING == 8;
                FLASHRNN_FORWARD_WARP_TILING_DIM_BATCH * FLASHRNN_FORWARD_WARP_TILING_DIM_GATE == 16 * 16 ;
                FLASHRNN_BACKWARD_WARP_TILING_DIM_BATCH * FLASHRNN_BACKWARD_WARP_TILING_DIM_HIDDEN == 16 * 16 ;

                # Sizes given by user
                HEAD_DIM == {config.hidden_dim // config.num_heads};
                NUM_HEADS == {config.num_heads};
                BATCH_DIM == {round_to_multiple(config.batch_size, 8 if config.dtype != 'float32' else 16)};
                FLASHRNN_HIDDEN_DIM == {config.hidden_dim};
                FLASHRNN_NUM_HEADS == {config.num_heads};

                # manual

                # this may be replaced by an inequality, to get a solution, but is slower
                # INTERNAL_HEAD_DIM is used internally
                HEAD_DIM == INTERNAL_HEAD_DIM ;
                FLASHRNN_FORWARD_SHARED_MEMORY_PADDING == SHARED_MEMORY_PADDING ;
                FLASHRNN_BACKWARD_SHARED_MEMORY_PADDING == SHARED_MEMORY_PADDING ;
                HEAD_DIM_SQ == HEAD_DIM ^ 2 ;
                # need this to match registers
                FLASHRNN_FORWARD_WARP_LOOPING_COUNT_BATCH <= 4 ;
                FLASHRNN_BACKWARD_WARP_LOOPING_COUNT_BATCH <= 4 ;
                FORWARD_FULL_COUNT_BATCH ==
                FLASHRNN_FORWARD_BLOCK_TILING_COUNT_BATCH * FLASHRNN_FORWARD_WARP_TILING_COUNT_BATCH * FLASHRNN_FORWARD_WARP_LOOPING_COUNT_BATCH;
                BACKWARD_FULL_COUNT_BATCH ==
                FLASHRNN_BACKWARD_BLOCK_TILING_COUNT_BATCH * FLASHRNN_BACKWARD_WARP_TILING_COUNT_BATCH * FLASHRNN_BACKWARD_WARP_LOOPING_COUNT_BATCH;

                # FORWARD PART
                FLASHRNN_FORWARD_WARP_TILING_COUNT_HIDDEN * FLASHRNN_FORWARD_WARP_TILING_COUNT_BATCH * HEAD_DIM * FLASHRNN_NUM_GATES_R
                == FORWARD_NUM_WARPS * FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE
                * FLASHRNN_FORWARD_WARP_LOOPING_COUNT_GATE * FLASHRNN_FORWARD_WARP_TILING_DIM_GATE ;
                HEAD_DIM % FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN * FLASHRNN_FORWARD_WARP_TILING_COUNT_HIDDEN
                * FLASHRNN_FORWARD_WARP_TILING_DIM_HIDDEN == 0 ;
                FLASHRNN_NUM_GATES_R * HEAD_DIM % (FLASHRNN_FORWARD_WARP_LOOPING_COUNT_GATE
                * FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE * FLASHRNN_FORWARD_WARP_TILING_DIM_GATE) == 0 ;
                FORWARD_NUM_WARPS * WARP_SIZE <= MAX_THREADS_PER_BLOCK_FORWARD;

                BATCH_DIM % FORWARD_FULL_COUNT_BATCH * FLASHRNN_FORWARD_WARP_TILING_DIM_BATCH == 0;

                FLASHRNN_FORWARD_NUM_BLOCKS <= STREAMING_MULTIPROCESSORS ;

                FLASHRNN_FORWARD_NUM_BLOCKS == FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE
                * FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN * FLASHRNN_FORWARD_BLOCK_TILING_COUNT_BATCH
                * FLASHRNN_FORWARD_MULTIHEAD_TILING_COUNT ;

                NUM_HEADS % FLASHRNN_FORWARD_MULTIHEAD_TILING_COUNT == 0 ;

                # recurrent register memory - measured in counts of fp (2 bytes or 4 bytes)
                FLASHRNN_FORWARD_WARP_TILING_DIM_HIDDEN * FLASHRNN_FORWARD_WARP_TILING_DIM_GATE
                * FLASHRNN_FORWARD_WARP_RECURRENT_CACHED_HIDDEN * FORWARD_NUM_WARPS
                == FORWARD_RECURRENT_REGISTER_MEMORY * FLASHRNN_FORWARD_WARP_TILING_COUNT_BATCH;
                FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN * FLASHRNN_FORWARD_WARP_RECURRENT_CACHED_HIDDEN
                * FLASHRNN_FORWARD_WARP_TILING_COUNT_HIDDEN * FLASHRNN_FORWARD_WARP_TILING_DIM_HIDDEN
                <= INTERNAL_HEAD_DIM;
                FORWARD_RECURRENT_REGISTER_MEMORY
                + FORWARD_NUM_WARPS * FLASHRNN_FORWARD_WARP_TILING_DIM_BATCH * FLASHRNN_FORWARD_WARP_TILING_DIM_GATE
                * FLASHRNN_FORWARD_WARP_LOOPING_COUNT_BATCH * {config._internal_acc_dtype_size // config._internal_dtype_size}
                + FORWARD_NUM_WARPS * FLASHRNN_FORWARD_WARP_TILING_DIM_BATCH * FLASHRNN_FORWARD_WARP_TILING_DIM_HIDDEN
                <= REGISTERS_PER_BLOCK_FORWARD ;

                # Shared memory size constraint in bytes, add 64 to keep FORWARD_RECURRENT_SHARED_MEMORY non-zero
                {config._internal_dtype_size} * FLASHRNN_NUM_GATES_R * FLASHRNN_FORWARD_BLOCK_TILING_COUNT_BATCH * HEAD_DIM_SQ
                + {config._internal_dtype_size} * HEAD_DIM * FLASHRNN_NUM_GATES_R * SHARED_MEMORY_PADDING * FLASHRNN_FORWARD_BLOCK_TILING_COUNT_BATCH + 64
                == FORWARD_RECURRENT_SHARED_MEMORY
                + {config._internal_dtype_size} * FORWARD_RECURRENT_REGISTER_MEMORY * FLASHRNN_FORWARD_BLOCK_TILING_COUNT_BATCH
                * FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE * FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN;

                # Memory for matrix multiplication results and aggregation
                {config._internal_acc_dtype_size} * FLASHRNN_FORWARD_WARP_TILING_COUNT_HIDDEN * FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN
                * FLASHRNN_NUM_GATES_R * FORWARD_FULL_COUNT_BATCH * FLASHRNN_FORWARD_WARP_TILING_DIM_BATCH * INTERNAL_HEAD_DIM
                + {config._internal_acc_dtype_size} * FLASHRNN_FORWARD_WARP_TILING_COUNT_HIDDEN * FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_GATE * FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN
                * FORWARD_FULL_COUNT_BATCH * FLASHRNN_FORWARD_WARP_TILING_DIM_BATCH * SHARED_MEMORY_PADDING == FORWARD_STATE_SHARED_MEMORY ;

                # total shared memory
                FORWARD_RECURRENT_SHARED_MEMORY + FORWARD_STATE_SHARED_MEMORY == FLASHRNN_FORWARD_SHARED_MEMORY_PER_HEAD_USED;
                FLASHRNN_FORWARD_MULTIHEAD_TILING_COUNT * FLASHRNN_FORWARD_SHARED_MEMORY_PER_HEAD_USED
                <= SHARED_MEMORY_PER_BLOCK * FLASHRNN_FORWARD_NUM_BLOCKS;

                # BACKWARD PART
                FLASHRNN_BACKWARD_WARP_TILING_COUNT_GATE * FLASHRNN_BACKWARD_WARP_TILING_COUNT_BATCH * INTERNAL_HEAD_DIM
                == BACKWARD_NUM_WARPS * FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_HIDDEN
                * FLASHRNN_BACKWARD_WARP_LOOPING_COUNT_HIDDEN * FLASHRNN_BACKWARD_WARP_TILING_DIM_HIDDEN ;
                HEAD_DIM % ( FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_HIDDEN * FLASHRNN_BACKWARD_WARP_TILING_DIM_HIDDEN
                * FLASHRNN_BACKWARD_WARP_LOOPING_COUNT_HIDDEN ) == 0 ;
                FLASHRNN_NUM_GATES_R * HEAD_DIM % (FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_GATE
                * FLASHRNN_BACKWARD_WARP_TILING_COUNT_GATE * FLASHRNN_BACKWARD_WARP_TILING_DIM_GATE) == 0;
                BACKWARD_NUM_WARPS * WARP_SIZE <= MAX_THREADS_PER_BLOCK_BACKWARD;

                BATCH_DIM % BACKWARD_FULL_COUNT_BATCH * FLASHRNN_BACKWARD_WARP_TILING_DIM_BATCH == 0;

                FLASHRNN_BACKWARD_NUM_BLOCKS <= STREAMING_MULTIPROCESSORS ;

                FLASHRNN_BACKWARD_NUM_BLOCKS == FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_GATE
                * FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_HIDDEN * FLASHRNN_BACKWARD_BLOCK_TILING_COUNT_BATCH
                * FLASHRNN_BACKWARD_MULTIHEAD_TILING_COUNT;

                NUM_HEADS % FLASHRNN_BACKWARD_MULTIHEAD_TILING_COUNT == 0 ;

                # recurrent register memory in dtypes (2 or 4 bytes)
                FLASHRNN_BACKWARD_WARP_TILING_DIM_HIDDEN * FLASHRNN_BACKWARD_WARP_TILING_DIM_GATE
                * FLASHRNN_BACKWARD_WARP_RECURRENT_CACHED_GATE * BACKWARD_NUM_WARPS
                == BACKWARD_RECURRENT_REGISTER_MEMORY * FLASHRNN_BACKWARD_WARP_TILING_COUNT_BATCH;
                FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_GATE * FLASHRNN_BACKWARD_WARP_RECURRENT_CACHED_GATE
                * FLASHRNN_BACKWARD_WARP_TILING_COUNT_GATE * FLASHRNN_BACKWARD_WARP_TILING_DIM_GATE
                <= FLASHRNN_NUM_GATES_R * INTERNAL_HEAD_DIM;
                BACKWARD_RECURRENT_REGISTER_MEMORY
                + BACKWARD_NUM_WARPS * FLASHRNN_BACKWARD_WARP_TILING_DIM_BATCH * FLASHRNN_BACKWARD_WARP_TILING_DIM_HIDDEN
                * FLASHRNN_BACKWARD_WARP_LOOPING_COUNT_BATCH * {config._internal_acc_dtype_size // config._internal_dtype_size}
                + BACKWARD_NUM_WARPS * FLASHRNN_BACKWARD_WARP_TILING_DIM_BATCH * FLASHRNN_BACKWARD_WARP_TILING_DIM_GATE
                <= REGISTERS_PER_BLOCK_BACKWARD ;

                # Shared memory size constraint in bytes, add 64 to keep BACKWARD_RECURRENT_SHARED_MEMORY non-zero
                {config._internal_dtype_size} * FLASHRNN_NUM_GATES_R * FLASHRNN_BACKWARD_BLOCK_TILING_COUNT_BATCH * HEAD_DIM_SQ
                + {config._internal_dtype_size} * HEAD_DIM * SHARED_MEMORY_PADDING * FLASHRNN_BACKWARD_BLOCK_TILING_COUNT_BATCH + 64
                == BACKWARD_RECURRENT_SHARED_MEMORY + {config._internal_dtype_size} * BACKWARD_RECURRENT_REGISTER_MEMORY * FLASHRNN_BACKWARD_BLOCK_TILING_COUNT_BATCH
                * FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_HIDDEN * FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_GATE ;

                {config._internal_acc_dtype_size} * FLASHRNN_BACKWARD_WARP_TILING_COUNT_GATE * BACKWARD_FULL_COUNT_BATCH
                * FLASHRNN_BACKWARD_WARP_TILING_DIM_BATCH * HEAD_DIM  * FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_GATE
                + {config._internal_acc_dtype_size} * FLASHRNN_BACKWARD_WARP_TILING_COUNT_GATE * FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_HIDDEN * FLASHRNN_BACKWARD_RECURRENT_TILING_COUNT_GATE
                * BACKWARD_FULL_COUNT_BATCH * FLASHRNN_BACKWARD_WARP_TILING_DIM_BATCH * SHARED_MEMORY_PADDING == BACKWARD_GATES_SHARED_MEMORY ;
                BACKWARD_RECURRENT_SHARED_MEMORY + BACKWARD_GATES_SHARED_MEMORY == FLASHRNN_BACKWARD_SHARED_MEMORY_PER_HEAD_USED;

                FLASHRNN_BACKWARD_MULTIHEAD_TILING_COUNT * FLASHRNN_BACKWARD_SHARED_MEMORY_PER_HEAD_USED <= SHARED_MEMORY_PER_BLOCK * FLASHRNN_BACKWARD_NUM_BLOCKS ;

            """
            )
            for constant, val in config.constants.items():
                if isinstance(val, int):
                    constraint_str += f"\n{constant} == {val};\n"
            name = config.function + "_f"
            sources = [
                str(curdir / "fused" / "flashrnn.cc"),
                str(curdir / "fused" / "flashrnn_fused_forward.cu"),
                str(curdir / "fused" / "flashrnn_fused_backward.cu"),
                str(curdir / "fused" / "flashrnn_fused_backward_cut.cu"),
                str(curdir / "util" / "blas.cu"),
                str(curdir / "util" / "cuda_error.cu"),
            ]
            seq_len = 2
            module = load_parametric_and_test_and_bisect(
                name=name,
                sources=sources,
                constraint_str=constraint_str,
                value_refinements=value_refinements,
                model_class="FlashRNNFuncFused",
                model_args=(
                    True,
                    config.batch_size,
                    config.hidden_dim,
                    config.num_heads,
                ),
                test_input=(
                    True,
                    torch.zeros(
                        [
                            seq_len,
                            config.batch_size,
                            config.num_heads,
                            config.head_dim,
                            config.num_gates_w,
                        ],
                        dtype=config.torch_dtype_w,
                        device="cuda",
                    ),
                    torch.zeros(
                        [
                            config.num_states,
                            config.batch_size,
                            config.num_heads,
                            config.head_dim,
                        ],
                        dtype=config.torch_dtype_s,
                        device="cuda",
                    ),
                    torch.zeros(
                        [
                            config.num_heads,
                            config.head_dim,
                            config.num_gates_r,
                            config.head_dim,
                        ],
                        dtype=config.torch_dtype_r,
                        device="cuda",
                    ),
                    torch.zeros(
                        [config.num_heads, config.head_dim, config.num_gates_t],
                        dtype=config.torch_dtype_b,
                        device="cuda",
                    ),
                ),
                value_to_independently_bisect_upwards_forward="FLASHRNN_FORWARD_WARP_RECURRENT_CACHED_HIDDEN",
                value_to_independently_bisect_upwards_backward="FLASHRNN_BACKWARD_WARP_RECURRENT_CACHED_GATE",
                map_output_to_backward_input=lambda test_input, output: (
                    test_input[3].permute(0, 3, 1, 2).contiguous(),
                    test_input[4].contiguous(),
                    output[0],
                    output[1],
                    output[2],
                    torch.ones_like(output[0]),
                ),
                extra_cflags=config.defines,
                extra_cuda_cflags=[
                    "-include",
                    str(curdir / "fused" / f"{config.function}_fused_pointwise.cuh"),
                ],
            )
            cls.mod[cfgdevstr] = module.FlashRNNFuncFused(
                True,
                config.batch_size,
                config.hidden_dim,
                config.num_heads,
            )
        return cls.mod[cfgdevstr]


def FlashRNNFuncGeneratorFused(training, config: FlashRNNConfig):
    flashrnn_cuda = _FlashRNNCUDAFused.instance(config)
    # pad batch size to multiple of 8
    round_batch_size = (
        round_to_multiple(config.batch_size, 8 if config.dtype != "float32" else 16)
        if config.dtype != "float32"
        else 16
    )

    class FlashRNNFunctionFused(torch.autograd.Function):
        @staticmethod
        @conditional_decorator(
            config.enable_automatic_mixed_precision,
            torch.amp.custom_fwd(
                device_type="cuda", cast_inputs=DTYPE_DICT[config.dtype]
            ),
        )
        def forward(ctx, training, *inputs):
            # pad input and state for batch size multiple of 8
            seq, bs, nheads, head_dim, gates_w = inputs[0].shape
            if bs % round_batch_size != 0:
                with torch.no_grad():
                    nstat, bs, nheads, head_dim = inputs[1].shape
                    bs_pad = round_batch_size * (
                        (bs + round_batch_size - 1) // round_batch_size
                    )
                    inp = torch.ones(
                        (seq, bs_pad, nheads, head_dim, gates_w),
                        dtype=inputs[0].dtype,
                        device=inputs[0].device,
                    )
                    stat = torch.ones(
                        (nstat, bs_pad, nheads, head_dim),
                        dtype=inputs[0].dtype,
                        device=inputs[0].device,
                    )
                    inp[:, :bs] = inputs[0][:, :bs]
                    stat[:, :bs] = inputs[1][:, :bs]
                    inputs = (inp, stat, inputs[2], inputs[3])
            inputs = (
                inputs[0].to(dtype=config.torch_dtype_w).contiguous(),
                inputs[1].to(dtype=config.torch_dtype_s).contiguous(),
                inputs[2].to(dtype=config.torch_dtype_r).contiguous(),
                inputs[3].to(dtype=config.torch_dtype_b).contiguous(),
            )
            states, cache_g_r, cache_g_i = flashrnn_cuda.forward(training, *inputs)
            if bs % round_batch_size != 0:
                states = states[:, :, :bs]
                cache_g_r = cache_g_r[:, :bs]
                if len(cache_g_i.shape) > 0:
                    cache_g_i = cache_g_i[:, :bs]
            ctx.save_for_backward(*inputs[2:], states, cache_g_r, cache_g_i)
            ctx.training = training
            return states

        @staticmethod
        @conditional_decorator(
            config.enable_automatic_mixed_precision,
            torch.amp.custom_bwd(device_type="cuda"),
        )
        def backward(ctx, states_grads):
            if not ctx.training:
                raise RuntimeError(
                    "FLASHRNN backward can only be called in training mode"
                )

            saved = [*ctx.saved_tensors]
            saved[0] = saved[0].permute(0, 3, 1, 2).contiguous()  # recurrent_kernel
            nstates, seq, bs, nheads, head_dim = states_grads.shape
            if bs % round_batch_size != 0:
                with torch.no_grad():
                    bs_pad = round_batch_size * (
                        (bs + round_batch_size - 1) // round_batch_size
                    )
                    states_grads_pad = torch.zeros(
                        (nstates, seq, bs_pad, nheads, head_dim),
                        dtype=states_grads.dtype,
                        device=states_grads.device,
                    )
                    states_grads_pad[:, :, :bs] = states_grads[:, :, :bs]
                    states_grads = states_grads_pad
                    states_pad = torch.ones(
                        (config.num_states, seq, bs_pad, nheads, head_dim),
                        dtype=saved[2].dtype,
                        device=saved[2].device,
                    )
                    states_pad[:, :, :bs] = saved[2][:, :, :bs]
                    cache_g_r_pad = torch.zeros(
                        (seq - 1, bs_pad, nheads, head_dim, saved[0].shape[3]),
                        dtype=saved[0].dtype,
                        device=saved[0].device,
                    )
                    cache_g_r_pad[:, :bs] = saved[3][:, :bs]
                    saved[2] = states_pad
                    saved[3] = cache_g_r_pad

                    if len(saved[4].shape) > 0:
                        cache_g_i_pad = torch.zeros(
                            (seq - 1, bs_pad, nheads, head_dim, saved[1]),
                            dtype=saved[4].dtype,
                            device=saved[4].device,
                        )
                        cache_g_i_pad[:, :bs] = saved[4][:, :bs]
                        saved[4] = cache_g_i_pad
            if config.gradient_recurrent_cut:
                grads = flashrnn_cuda.backward_cut(*saved, states_grads.contiguous())
            else:
                grads = flashrnn_cuda.backward(*saved, states_grads.contiguous())
            with torch.no_grad():
                S, B, nheads, head_dim, _ = grads[0].shape
                if config.num_gates_w != config.num_gates_t:
                    grads[0] = (
                        grads[0]
                        .view(S, B, nheads, head_dim, config.num_gates_i)[
                            :, :, :, :, (config.num_gates_i - config.num_gates_w) :
                        ]
                        .reshape(S, B, -1)
                    )
                if bs % round_batch_size != 0:
                    grads[0] = grads[0][:, :bs]
                    grads[1] = grads[1][:, :bs]

            return (None, *grads)

    return FlashRNNFunctionFused


class _FlashRNNCUDA:
    mod = {}

    @classmethod
    def instance(cls, config: FlashRNNConfig):
        cfgdevstr = repr(config) + f"_{torch.cuda.current_device()}"
        if cfgdevstr not in cls.mod:
            module = load(
                name=config.function,
                sources=[
                    str(curdir / "alternating" / "flashrnn.cc"),
                    str(curdir / "alternating" / "flashrnn_forward.cu"),
                    str(curdir / "alternating" / "flashrnn_backward.cu"),
                    str(curdir / "alternating" / "flashrnn_backward_cut.cu"),
                    str(curdir / "alternating" / f"{config.function}_pointwise.cu"),
                    str(curdir / "util" / "blas.cu"),
                    str(curdir / "util" / "cuda_error.cu"),
                ],
                extra_cflags=[
                    f"-D{const}={constval}"
                    for const, constval in config.constants.items()
                ]
                + config.defines,
            )
            cls.mod[cfgdevstr] = module.FlashRNNFunc(
                True, config.batch_size, config.hidden_dim, config.num_heads
            )

        return cls.mod[cfgdevstr]


def FlashRNNFuncGenerator(training, config: FlashRNNConfig):
    flashrnn_cuda = _FlashRNNCUDA.instance(config=config)

    class FlashRNNFunction(torch.autograd.Function):
        @staticmethod
        @conditional_decorator(
            config.enable_automatic_mixed_precision,
            torch.amp.custom_fwd(
                device_type="cuda", cast_inputs=DTYPE_DICT[config.dtype]
            ),
        )
        def forward(ctx, training, *inputs):
            if config.enable_automatic_mixed_precision:
                inputs = (
                    inputs[0].to(dtype=config.torch_dtype_w),
                    inputs[1].to(dtype=config.torch_dtype_s),
                    inputs[2].to(dtype=config.torch_dtype_r),
                    inputs[3].to(dtype=config.torch_dtype_b),
                )
            states, cache_g_r, cache_g_i = flashrnn_cuda.forward(training, *inputs)

            ctx.save_for_backward(*inputs[2:], states, cache_g_r, cache_g_i)
            ctx.training = training
            return states

        @staticmethod
        @once_differentiable
        @conditional_decorator(
            config.enable_automatic_mixed_precision,
            torch.amp.custom_bwd(device_type="cuda"),
        )
        def backward(ctx, grad_s):
            if not ctx.training:
                raise RuntimeError(
                    "FlashRNN backward can only be called in training mode"
                )
            saved = [*ctx.saved_tensors]
            saved[0] = saved[0].permute(0, 2, 3, 1).contiguous()  # transpose R
            if config.gradient_recurrent_cut:
                grads = flashrnn_cuda.backward_cut(*saved, grad_s.contiguous())
            else:
                grads = flashrnn_cuda.backward(*saved, grad_s.contiguous())
            with torch.no_grad():
                S, B, num_heads, wgates, head_dim = grads[0].shape
                if config.num_gates_w != config.num_gates_t:
                    wgrad = grads[0].view(S, B, num_heads, config.num_gates_i, head_dim)
                    wgrad = wgrad[:, :, :, (config.num_gates_i - config.num_gates_w) :]
                    grads[0] = wgrad.reshape(
                        S, B, num_heads, config.num_gates_w, head_dim
                    )
            return (None, *grads)

    return FlashRNNFunction


def _permute_input(config: FlashRNNConfig, x: torch.Tensor) -> torch.Tensor:
    if config._internal_input_permutation is None:
        return x
    else:
        return x.permute(config._internal_input_permutation)


def _permute_recurrent_weight(config: FlashRNNConfig, R: torch.Tensor) -> torch.Tensor:
    if config._internal_recurrent_permutation is None:
        return R
    else:
        return R.permute(config._internal_recurrent_permutation)


def _permute_bias(config: FlashRNNConfig, b: torch.Tensor) -> torch.Tensor:
    if config._internal_bias_permutation is None:
        return b
    else:
        return b.permute(config._internal_bias_permutation)


def _permute_output(config: FlashRNNConfig, x: torch.Tensor) -> torch.Tensor:
    if config._internal_output_permutation is None:
        return x
    else:
        return x.permute(config._internal_output_permutation)


def _permute_output_backward(config: FlashRNNConfig, x: torch.Tensor) -> torch.Tensor:
    if config._internal_output_backward_permutation is None:
        return x
    else:
        return x.permute(config._internal_output_backward_permutation)


def _zero_state(config: FlashRNNConfig, inp: torch.Tensor) -> torch.Tensor:
    """Returns a nested structure of zero Tensors with the same structure
    and shape as []. The returned Tensors will have the same
    dtype and be on the same device as `inp`.

    Arguments:
        inp: Tensor, to specify the device and dtype of the returned tensors.
        shape_state: nested structure of integers.

    Returns:
        zero_state: a nested structure of zero Tensors.

    Raises:
        ValueError: if `state_shape` has non-integer values.
    """
    batch_dim = inp.shape[config.input_shape.index("B")]
    state = torch.zeros(
        (config.num_states, batch_dim, config.num_heads, config.head_dim),
        dtype=inp.dtype,
        device=inp.device,
    )
    with torch.no_grad():
        if isinstance(config.initial_val, float):
            state += config.initial_val
        else:
            for i in range(config.num_states):
                state[i] += config.initial_val[i]
    return state[None, :].permute(permute_to("TSBHD", config.output_shape))


def _get_kernel_step(config: FlashRNNConfig):
    if config.backend == "vanilla":

        def fn(Wx, states, R, b, **kwargs):
            return flashrnn_forward_step(
                Wx,
                states,
                R,
                b,
                pointwise_forward=flashrnn_pointwise_function_registry[config.function],
                constants=config.constants,
                **kwargs,
            )

    elif config.backend == "cuda" or config.backend == "cuda_fused":

        def fn(Wx, states, R, b, **kwargs):
            states = FlashRNNFuncGenerator(
                torch.is_grad_enabled(), config=config
            ).apply(
                torch.is_grad_enabled(),
                Wx.contiguous(),
                states[:, 0].contiguous(),
                R.contiguous(),
                b.contiguous(),
            )
            return states[:, 1:], states[:, -1:]

    return fn


def _get_kernel(config: FlashRNNConfig):
    if config.backend == "vanilla":

        def fn(Wx, states, R, b, **kwargs):
            return flashrnn_forward(
                Wx,
                states,
                R,
                b,
                pointwise_forward=flashrnn_pointwise_function_registry[config.function],
                constants=config.constants,
                **kwargs,
            )

    elif config.backend == "vanilla_fwbw":
        if config.function == "lstm":
            from .vanilla_fwbw.fwbw import lstm_pt_fwbw

            def fn(Wx, states, R, b, **kwargs):
                return lstm_pt_fwbw(
                    states_initial=states,
                    Wx=Wx,
                    R=R,
                    b=b,
                    backward_recurrent_clip_val=config.gradient_recurrent_clipval,
                    autocast_kernel_dtype=config.dtype,
                )
        elif config.function == "slstm":
            from .vanilla_fwbw.fwbw import slstm_pt_fwbw

            def fn(Wx, states, R, b, **kwargs):
                return slstm_pt_fwbw(
                    states_initial=states,
                    Wx=Wx,
                    R=R,
                    b=b,
                    backward_recurrent_clip_val=config.gradient_recurrent_clipval,
                    autocast_kernel_dtype=config.dtype,
                )
        else:
            raise NotImplementedError(
                f"Function {config.function} not implemented for vanilla_fwbw backend."
            )

    elif config.backend == "cuda":

        def fn(Wx, states, R, b, **kwargs):
            states = FlashRNNFuncGenerator(
                torch.is_grad_enabled(), config=config
            ).apply(
                torch.is_grad_enabled(),
                Wx.contiguous(),
                states[:, 0].contiguous(),
                R.contiguous(),
                b.contiguous(),
            )
            return states[:, 1:], states[:, -1:]

    elif config.backend == "cuda_fused":

        def fn(Wx, states, R, b, **kwargs):
            states = FlashRNNFuncGeneratorFused(
                torch.is_grad_enabled(), config=config
            ).apply(
                torch.is_grad_enabled(),
                Wx.contiguous(),
                states[:, 0].contiguous(),
                R.contiguous(),
                b.contiguous(),
            )
            return states[:, 1:], states[:, -1:]

    elif config.backend == "triton_fused":
        if config.function == "lstm":
            from .triton_fused.fwbw import lstm_tr_fwbw

            def fn(Wx, states, R, b, **kwargs):
                return lstm_tr_fwbw(
                    states_initial=states,
                    Wx=Wx,
                    R=R,
                    b=b,
                    backward_recurrent_clip_val=config.gradient_recurrent_clipval,
                    autocast_kernel_dtype=config.dtype,
                )

        elif config.function == "slstm":
            from .triton_fused.fwbw import slstm_tr_fwbw

            def fn(Wx, states, R, b, **kwargs):
                return slstm_tr_fwbw(
                    states_initial=states,
                    Wx=Wx,
                    R=R,
                    b=b,
                    backward_recurrent_clip_val=config.gradient_recurrent_clipval,
                    autocast_kernel_dtype=config.dtype,
                )

    else:
        raise ValueError(f"Unknown backend {config.backend}")

    return fn


def _get_config(
    Wx: torch.Tensor,
    R: torch.Tensor,
    b: torch.Tensor,
    function: str,
    backend: str,
    dtype: Optional[str],
) -> FlashRNNConfig:
    return FlashRNNConfig(
        head_dim=Wx.shape[4],
        num_heads=Wx.shape[3],
        batch_size=Wx.shape[0],
        function=function,
        backend=backend,
        dtype=dtype if dtype is not None else "bfloat16",
        dtype_w=DTYPE_DICT_REV[Wx.dtype],
        dtype_r=DTYPE_DICT_REV[R.dtype],
        dtype_b=DTYPE_DICT_REV[b.dtype],
    )


def flashrnn(
    Wx: torch.Tensor,
    R: torch.Tensor,
    b: torch.Tensor,
    states: Optional[torch.Tensor] = None,
    function: str = "lstm",
    config: Optional[FlashRNNConfig] = None,
    backend: str = "cuda_fused",
    dtype: str = "bfloat16",
):
    if config is None:
        config = _get_config(Wx, R, b, function, backend, dtype=dtype)

    kernel = _get_kernel(config)
    if states is None:
        states = _zero_state(config, Wx)
    states = _permute_output_backward(config, states)
    Wx = _permute_input(config, Wx)
    R = _permute_recurrent_weight(config, R)
    b = _permute_bias(config, b)
    h, last_h = kernel(Wx, states, R, b)
    return _permute_output(config, h), _permute_output(config, last_h)
