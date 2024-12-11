import torch
import triton
from dataclasses import dataclass
from typing import Literal, Callable
from flashrnn.flashrnn import flashrnn
from torch import nn
from haste_pytorch import LSTM as LSTM_haste

"""Benchmarks different kernels.

Enter in line vals the different kernels you want to benchmark.

Convention: <function>--<backend>++<fw|fwbw>++<[compile]>

fw: forward pass only
fwbw: forward and backward pass
compile: apply torch.compile
"""

# #! Parameters
# BENCHMARK_NAME = "flashrnnspeedbench"
# B, N_HEADS = 1, 4
# HEAD_DIMS = [64, 128, 256, 512]
# DTYPE = "bfloat16"

# WARMUP = 25
# REP = 1000
# #! =================

_flashrnn_function_to_num_gates = {
    "lstm": 4,
    "slstm": 4,
}


@dataclass
class KernelSpeedBenchmarkConfig:
    benchmark_name: str
    kernel_specifiers: list[str]
    warmup: int = 500
    rep: int = 2000
    dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"


# we repeat colors to have enough colors for all the kernels
colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    # repeat colors
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    # repeat colors
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
linestyle_mapping = {"fw": "--", "fwbw": "-"}


def create_kernel2style_mapping(kernel_names: list[str]) -> list[tuple[str, str]]:
    """This function maps a kernel name (as specified by the convention) to a color and a linestyle."""
    raw_kernel_names = [kernel_name.split("++")[0] for kernel_name in kernel_names]
    raw_kernel_names = list(set(raw_kernel_names))
    assert len(raw_kernel_names) <= len(colors), "Not enough colors for all the kernels"
    # map kernel name to color
    kernel2color = {
        kernel_name: color for kernel_name, color in zip(raw_kernel_names, colors)
    }
    # map kernel name to style
    kernel_names2style = []
    for kernel_name in kernel_names:
        raw_kernel_name = kernel_name.split("++")[0]
        fwbw_type = kernel_name.split("++")[1]
        kernel_names2style.append(
            (kernel2color[raw_kernel_name], linestyle_mapping[fwbw_type])
        )
    return kernel_names2style


def create_head_dimension_configs(
    benchmark_config: KernelSpeedBenchmarkConfig,
) -> list[triton.testing.Benchmark]:
    configs = []
    kernels_to_benchmark = benchmark_config.kernel_specifiers
    B = 16
    T = 1024  # 256
    configs.append(
        triton.testing.Benchmark(
            x_names=["DH", "NH"],
            x_vals=[(16, 48), (32, 24), (64, 12), (128, 6), (256, 3), (768, 1)],
            line_arg="provider",
            line_vals=kernels_to_benchmark,
            line_names=kernels_to_benchmark,
            styles=create_kernel2style_mapping(kernels_to_benchmark),
            ylabel="ms",
            plot_name=f"{benchmark_config.benchmark_name}--batch-{B}--T-{T}--dtype-{benchmark_config.dtype}",
            args={
                "B": B,
                "T": T,
            },
        )
    )
    return configs


def create_batch_size_configs(
    benchmark_config: KernelSpeedBenchmarkConfig,
    dh_nh_pairs: list[tuple[int, int]] = [(64, 12), (768, 1)],
) -> list[triton.testing.Benchmark]:
    """
    We vary the batch size and use these (DH, NH) pairs:
    (64, 12), (768, 1)
    Note in this experiment cuda_fused can be faster for smaller batch sizes
    if the batch_size in the config is set to 8 instead of 16.
    """
    configs = []
    kernels_to_benchmark = benchmark_config.kernel_specifiers
    T = 1024  # 256
    for DH, NH in dh_nh_pairs:
        configs.append(
            triton.testing.Benchmark(
                x_names=["B"],
                x_vals=[2, 8, 16, 32, 64, 128, 256],
                line_arg="provider",
                line_vals=kernels_to_benchmark,
                line_names=kernels_to_benchmark,
                styles=create_kernel2style_mapping(kernels_to_benchmark),
                ylabel="ms",
                plot_name=f"{benchmark_config.benchmark_name}--dh-{DH}--nh-{NH}--T-{T}--dtype-{benchmark_config.dtype}",
                args={
                    "NH": NH,
                    "DH": DH,
                    "T": T,
                },
            )
        )
    return configs


def create_sequence_length_configs(
    benchmark_config: KernelSpeedBenchmarkConfig,
) -> list[triton.testing.Benchmark]:
    """
    We vary the sequence length and use these (DH, NH) pairs:
    (64, 12), (768, 1)
    """
    configs = []
    kernels_to_benchmark = benchmark_config.kernel_specifiers
    B = 16
    dh_nh_pairs = [(64, 12), (768, 1)]
    for DH, NH in dh_nh_pairs:
        configs.append(
            triton.testing.Benchmark(
                x_names=["T"],
                x_vals=[256, 512, 1024, 2048],
                line_arg="provider",
                line_vals=kernels_to_benchmark,
                line_names=kernels_to_benchmark,
                styles=create_kernel2style_mapping(kernels_to_benchmark),
                ylabel="ms",
                plot_name=f"{benchmark_config.benchmark_name}--dh-{DH}--nh-{NH}--B-{B}--dtype-{benchmark_config.dtype}",
                args={
                    "NH": NH,
                    "DH": DH,
                    "B": B,
                },
            )
        )
    return configs


@dataclass
class KernelSpec:
    function: str
    backend: str
    fwbw: bool
    use_torch_compile: bool

    @staticmethod
    def parse_from_string(kernel_specifier: str):
        parts_minus = kernel_specifier.split("--")
        parts_plus = parts_minus[1].split("++")
        function = parts_minus[0]
        backend = parts_plus[0]
        fwbw = "fwbw" in parts_plus[1]
        if len(parts_plus) > 2:
            use_torch_compile = "compile" in parts_plus[2]
        else:
            use_torch_compile = False

        return KernelSpec(function, backend, fwbw, use_torch_compile)


def get_flashrnn_kernel_benchmark_fn(kernel_spec: KernelSpec) -> Callable:
    def kernel_fn(
        Wx: torch.Tensor,
        R: torch.Tensor,
        b: torch.Tensor,
        dtype: str,
        gate_linear: nn.Module = None,
        x_only: torch.Tensor = None,
    ):
        if kernel_spec.use_torch_compile:
            flashrnn_fn = torch.compile(flashrnn)
        else:
            flashrnn_fn = flashrnn

        if gate_linear is not None:
            Wx = gate_linear(x_only)
            Wx = Wx.reshape(
                Wx.shape[0], Wx.shape[1], R.shape[0], R.shape[1], R.shape[2]
            )

        h_frnn, hlast_frnn = flashrnn_fn(
            Wx=Wx,
            R=R,
            b=b,
            states=None,
            function=kernel_spec.function,
            backend=kernel_spec.backend,
            dtype=dtype,
        )
        if kernel_spec.fwbw:
            # run the backward pass
            h_frnn[0].sum().backward()

    return kernel_fn


def get_runnable_benchmark(
    run_configs: list[triton.testing.Benchmark],
    benchmark_config: KernelSpeedBenchmarkConfig,
):
    @triton.testing.perf_report(run_configs)
    def bench_flashrnn(
        B: int,
        NH: int,
        T: int,
        DH: int,
        provider: str,
        bench_config: KernelSpeedBenchmarkConfig = benchmark_config,
        device: str = "cuda",
    ):
        dtype = getattr(torch, bench_config.dtype)

        # select kernel
        kernel_spec = KernelSpec.parse_from_string(kernel_specifier=provider)

        requires_grad = (
            kernel_spec.fwbw
        )  # if we are running the backward pass, we need to compute the gradients

        if kernel_spec.function == "attention_causal":
            from baseline_kernels.torch_sdp_attention import (
                get_attention_causal_pt_kernel,
            )

            q = (
                torch.randn([B, NH, T, DH], device=device, dtype=dtype)
                .clone()
                .detach()
                .requires_grad_(requires_grad)
            )
            k = (
                torch.randn([B, NH, T, DH], device=device, dtype=dtype)
                .clone()
                .detach()
                .requires_grad_(requires_grad)
            )
            v = (
                torch.randn([B, NH, T, DH], device=device, dtype=dtype)
                .clone()
                .detach()
                .requires_grad_(requires_grad)
            )

            attention_kernel = get_attention_causal_pt_kernel(kernel_spec.backend)

            def run_kernel_fn():
                out = attention_kernel(q, k, v)
                if kernel_spec.fwbw:
                    out.sum().backward()

        elif kernel_spec.function == "nn.LSTM":
            nn_lstm_dtype_str = kernel_spec.backend.split("-")[-1]
            assert nn_lstm_dtype_str in [
                "bfloat16",
                "float32",
                "float16",
            ], f"Invalid dtype for nn.LSTM, got {nn_lstm_dtype_str}"
            nn_lstm_dtype = getattr(torch, nn_lstm_dtype_str)
            torch_lstm = torch.nn.LSTM(
                DH, DH, 1, bias=True, batch_first=True, bidirectional=False
            ).to(device=device, dtype=nn_lstm_dtype)

            pt_in = (
                torch.randn([B, T, DH], device=device, dtype=nn_lstm_dtype)
                .clone()
                .detach()
                .requires_grad_(True)
            )

            def run_kernel_fn():
                out = torch_lstm(pt_in)
                if kernel_spec.fwbw:
                    out[0].sum().backward()

        elif kernel_spec.function == "haste.LSTM":
            nn_lstm_dtype_str = kernel_spec.backend.split("-")[-1]
            assert nn_lstm_dtype_str in [
                "float32",
            ], f"Invalid dtype for haste.LSTM, got {nn_lstm_dtype_str}"
            nn_lstm_dtype = getattr(torch, nn_lstm_dtype_str)
            haste_lstm = LSTM_haste(DH, DH, batch_first=True).to(
                device=device, dtype=nn_lstm_dtype
            )

            pt_in = (
                torch.randn([B, T, DH], device=device, dtype=nn_lstm_dtype)
                .clone()
                .detach()
                .requires_grad_(True)
            )

            def run_kernel_fn():
                out = haste_lstm(pt_in)
                if kernel_spec.fwbw:
                    out[0].sum().backward()
        else:
            num_gates = _flashrnn_function_to_num_gates[kernel_spec.function]
            # create input tensors
            Wx = torch.randn(
                [B, T, num_gates, NH, DH],
                device=device,
                dtype=dtype,
                requires_grad=requires_grad,
            )
            x_only = torch.randn(
                [B, T, NH * DH],
                device=device,
                dtype=dtype,
                requires_grad=requires_grad,
            )
            R = torch.randn(
                [num_gates, NH, DH, DH],
                device=device,
                dtype=dtype,
                requires_grad=requires_grad,
            ) / (DH**0.5)
            b = torch.randn(
                [num_gates, NH, DH],
                device=device,
                dtype=dtype,
                requires_grad=requires_grad,
            )

            Wx_mtr = Wx.clone().to(dtype=dtype).detach().requires_grad_(requires_grad)
            R_mtr = R.clone().to(dtype=dtype).detach().requires_grad_(requires_grad)
            b_mtr = b.clone().to(dtype=dtype).detach().requires_grad_(requires_grad)
            x_only_mtr = (
                x_only.clone().to(dtype=dtype).detach().requires_grad_(requires_grad)
            )

            # check if we need to add a torch nn.Linear layer to compute the gate preactivations
            kernel_backend_split = kernel_spec.backend.split("-")
            if len(kernel_backend_split) > 1:
                assert len(kernel_backend_split) == 2, "Invalid kernel backend"
                backend_name = kernel_backend_split[0]
                if "withlinear" in kernel_backend_split[1]:
                    with_linear = True
                else:
                    with_linear = False
            else:
                backend_name = kernel_spec.backend
                with_linear = False

            kernel_spec.backend = backend_name

            if with_linear:
                gate_linear = torch.nn.Linear(NH * DH, num_gates * NH * DH).to(
                    device=device, dtype=dtype
                )
            else:
                gate_linear = None

            # get the benchmark function
            kernel_benchmark_fn = get_flashrnn_kernel_benchmark_fn(kernel_spec)

            def run_kernel_fn():
                kernel_benchmark_fn(
                    Wx_mtr,
                    R_mtr,
                    b_mtr,
                    bench_config.dtype,
                    gate_linear=gate_linear,
                    x_only=x_only_mtr,
                )

        print(
            f"[NEW CONFIGURATION] Running speedtest for {provider}, with batch size {B}, num heads {NH}, context size {T}, head dim {DH}, dtype {bench_config.dtype}"
        )
        try:
            ms = triton.testing.do_bench(
                run_kernel_fn, warmup=bench_config.warmup, rep=bench_config.rep
            )
        except Exception as e:
            print(f"Error: {e}")
            ms = float("nan")
        return ms

    return bench_flashrnn


def paperplot_experiments():
    OUTPUT_DIR = "./outputs_speed_exps_v5"
    ### head dimension experiment
    print("====================================")
    print("HEAD DIMENSION EXPERIMENT")
    print("====================================")
    head_dim_benchmark_config = KernelSpeedBenchmarkConfig(
        benchmark_name="head_dimension_exp",
        kernel_specifiers=[
            ## lstm
            # fw
            "lstm--vanilla++fw",
            "lstm--vanilla_fwbw++fw",
            "lstm--triton_fused++fw",
            "lstm--cuda_fused++fw",
            "lstm--cuda++fw",
            # fwbw
            "lstm--vanilla_fwbw++fwbw",
            "lstm--vanilla++fwbw",
            "lstm--triton_fused++fwbw",
            "lstm--cuda_fused++fwbw",
            "lstm--cuda++fwbw",
            ## slstm
            # fw
            "slstm--vanilla++fw",
            "slstm--vanilla_fwbw++fw",
            "slstm--triton_fused++fw",
            "slstm--cuda_fused++fw",
            "slstm--cuda++fw",
            # fwbw
            "slstm--vanilla_fwbw++fwbw",
            "slstm--triton_fused++fwbw",
            "slstm--cuda_fused++fwbw",
            "slstm--cuda++fwbw",
            ## baselines
            "nn.LSTM--pytorch-bfloat16++fw",
            "nn.LSTM--pytorch-bfloat16++fwbw",
            "nn.LSTM--pytorch-float32++fw",
            "nn.LSTM--pytorch-float32++fwbw",
            "nn.LSTM--pytorch-float16++fw",
            "nn.LSTM--pytorch-float16++fwbw",
            "haste.LSTM--pytorch-float32++fw",
            "haste.LSTM--pytorch-float32++fwbw",
            "attention_causal--fa2++fw",
            "attention_causal--fa2++fwbw",
            # "attention_causal--cudnn++fw",
            # "attention_causal--cudnn++fwbw",
            # "attention_causal--efficient++fw",
            # "attention_causal--efficient++fwbw",
            # "attention_causal--math++fw",
            # "attention_causal--math++fwbw",
        ],
        warmup=25,
        rep=1000,
        dtype="bfloat16",
    )

    head_dim_run_configs = create_head_dimension_configs(head_dim_benchmark_config)

    head_dimension_benchmark_fn = get_runnable_benchmark(
        head_dim_run_configs, head_dim_benchmark_config
    )

    head_dimension_benchmark_fn.run(
        save_path=f"{OUTPUT_DIR}/{head_dim_benchmark_config.benchmark_name}",
        print_data=True,
    )
    ### =================

    ### batch size experiment
    print("====================================")
    print("BATCH SIZE EXPERIMENT")
    print("====================================")
    batch_size_benchmark_config = KernelSpeedBenchmarkConfig(
        benchmark_name="batch_size_exp",
        kernel_specifiers=[
            ## lstm
            # fw
            # "lstm--vanilla++fw",
            "lstm--vanilla_fwbw++fw",
            "lstm--triton_fused++fw",
            "lstm--cuda_fused++fw",
            "lstm--cuda++fw",
            # fwbw
            "lstm--vanilla_fwbw++fwbw",
            "lstm--triton_fused++fwbw",
            "lstm--cuda_fused++fwbw",
            "lstm--cuda++fwbw",
            ## slstm
            # fw
            # "slstm--vanilla++fw",
            "slstm--vanilla_fwbw++fw",
            "slstm--triton_fused++fw",
            "slstm--cuda_fused++fw",
            "slstm--cuda++fw",
            # fwbw
            "slstm--vanilla_fwbw++fwbw",
            "slstm--triton_fused++fwbw",
            "slstm--cuda_fused++fwbw",
            "slstm--cuda++fwbw",
            ## baselines
            "nn.LSTM--pytorch-bfloat16++fw",
            "nn.LSTM--pytorch-bfloat16++fwbw",
            "nn.LSTM--pytorch-float32++fw",
            "nn.LSTM--pytorch-float32++fwbw",
            "nn.LSTM--pytorch-float16++fw",
            "nn.LSTM--pytorch-float16++fwbw",
            "attention_causal--fa2++fw",
            "attention_causal--fa2++fwbw",
            # "attention_causal--cudnn++fw",
            # "attention_causal--cudnn++fwbw",
            # "attention_causal--efficient++fw",
            # "attention_causal--efficient++fwbw",
            # "attention_causal--math++fw",
            # "attention_causal--math++fwbw",
        ],
        warmup=25,
        rep=1000,
        dtype="bfloat16",
    )

    batch_size_run_configs = create_batch_size_configs(batch_size_benchmark_config)

    batch_size_benchmark_fn = get_runnable_benchmark(
        batch_size_run_configs, batch_size_benchmark_config
    )

    batch_size_benchmark_fn.run(
        save_path=f"{OUTPUT_DIR}/{batch_size_benchmark_config.benchmark_name}",
        print_data=True,
    )
    ### =================

    ### sequence length experiment
    print("====================================")
    print("SEQUENCE LENGTH EXPERIMENT")
    print("====================================")
    sequence_length_benchmark_config = KernelSpeedBenchmarkConfig(
        benchmark_name="sequence_length_exp",
        kernel_specifiers=[
            ## lstm
            # fw
            # "lstm--vanilla++fw",
            "lstm--vanilla_fwbw++fw",
            "lstm--triton_fused++fw",
            "lstm--cuda_fused++fw",
            "lstm--cuda++fw",
            # fwbw
            "lstm--vanilla_fwbw++fwbw",
            "lstm--triton_fused++fwbw",
            "lstm--cuda_fused++fwbw",
            "lstm--cuda++fwbw",
            ## slstm
            # fw
            # "slstm--vanilla++fw",
            "slstm--vanilla_fwbw++fw",
            "slstm--triton_fused++fw",
            "slstm--cuda_fused++fw",
            "slstm--cuda++fw",
            # # fwbw
            "slstm--vanilla_fwbw++fwbw",
            "slstm--triton_fused++fwbw",
            "slstm--cuda_fused++fwbw",
            "slstm--cuda++fwbw",
            ## baselines
            "nn.LSTM--pytorch-bfloat16++fw",
            "nn.LSTM--pytorch-bfloat16++fwbw",
            "nn.LSTM--pytorch-float32++fw",
            "nn.LSTM--pytorch-float32++fwbw",
            "nn.LSTM--pytorch-float16++fw",
            "nn.LSTM--pytorch-float16++fwbw",
            "attention_causal--fa2++fw",
            "attention_causal--fa2++fwbw",
            # "attention_causal--cudnn++fw",
            # "attention_causal--cudnn++fwbw",
            # "attention_causal--efficient++fw",
            # "attention_causal--efficient++fwbw",
            # "attention_causal--math++fw",
            # "attention_causal--math++fwbw",
        ],
        warmup=25,
        rep=1000,
        dtype="bfloat16",
    )

    sequence_length_run_configs = create_sequence_length_configs(
        sequence_length_benchmark_config
    )

    sequence_length_benchmark_fn = get_runnable_benchmark(
        sequence_length_run_configs, sequence_length_benchmark_config
    )

    sequence_length_benchmark_fn.run(
        save_path=f"{OUTPUT_DIR}/{sequence_length_benchmark_config.benchmark_name}",
        print_data=True,
    )
    ### =================


def debug_experiments():
    OUTPUT_DIR = "./outputs_speed_exps_v2_debug6"
    ### head dimension experiment
    print("====================================")
    print("HEAD DIMENSION EXPERIMENT")
    print("====================================")
    head_dim_benchmark_config = KernelSpeedBenchmarkConfig(
        benchmark_name="head_dimension_exp",
        kernel_specifiers=[
            ## lstm
            # fw
            # "lstm--vanilla++fw",
            # "lstm--vanilla_fwbw++fw",
            # "lstm--triton_fused-withlinear++fw",
            "slstm--cuda_fused-withlinear++fw",
            "slstm--cuda_fused++fw",
            "slstm--cuda-withlinear++fw",
            # # fwbw
            # # "lstm--vanilla_fwbw++fwbw",
            # # "lstm--vanilla++fwbw",
            # "lstm--triton_fused++fwbw",
            "slstm--cuda_fused-withlinear++fwbw",
            "slstm--cuda_fused++fwbw",
            "slstm--cuda-withlinear++fwbw",
            # "lstm--cuda++fwbw",
            ## slstm
            # fw
            # "slstm--vanilla++fw",
            # "slstm--vanilla_fwbw++fw",
            # "slstm--triton_fused++fw",
            # "slstm--cuda_fused++fw",
            # "slstm--cuda++fw",
            # # fwbw
            # "slstm--vanilla_fwbw++fwbw",
            # "slstm--triton_fused++fwbw",
            # "slstm--cuda_fused++fwbw",
            # "slstm--cuda++fwbw",
            ## baselines
            # "haste.LSTM--pytorch-float32++fw",
            # "haste.LSTM--pytorch-float32++fwbw",
            # "nn.LSTM--pytorch-bfloat16++fw",
            # "nn.LSTM--pytorch-bfloat16++fwbw",
            # "nn.LSTM--pytorch-float32++fw",
            # "nn.LSTM--pytorch-float32++fwbw",
            # "nn.LSTM--pytorch-float16++fw",
            # "nn.LSTM--pytorch-float16++fwbw",
            # "attention_causal--fa2++fw",
            # "attention_causal--fa2++fwbw",
            # "attention_causal--cudnn++fw",
            # "attention_causal--cudnn++fwbw",
            # "attention_causal--efficient++fw",
            # "attention_causal--efficient++fwbw",
            # "attention_causal--math++fw",
            # "attention_causal--math++fwbw",
        ],
        warmup=40,
        rep=2000,
        dtype="bfloat16",  # "float32",
    )

    head_dim_run_configs = create_head_dimension_configs(head_dim_benchmark_config)

    head_dimension_benchmark_fn = get_runnable_benchmark(
        head_dim_run_configs, head_dim_benchmark_config
    )

    head_dimension_benchmark_fn.run(
        save_path=f"{OUTPUT_DIR}/{head_dim_benchmark_config.benchmark_name}",
        print_data=True,
    )
    ### =================


def paper_plot_experiments_additional():
    OUTPUT_DIR = "./outputs_speed_exps_add_v2"
    ### head dimension experiment
    print("====================================")
    print("BATCH SIZE ADDITIONAL EXPERIMENT")
    print("====================================")
    batch_size_add_benchmark_config = KernelSpeedBenchmarkConfig(
        benchmark_name="batch_size_exp_additional",
        kernel_specifiers=[
            ## lstm
            # fw
            # "lstm--vanilla++fw",
            # "lstm--vanilla_fwbw++fw",
            # "lstm--triton_fused++fw",
            "lstm--cuda_fused++fw",
            "lstm--cuda++fw",
            # "lstm--triton_fused-withlinear++fw",
            "lstm--cuda_fused-withlinear++fw",
            "lstm--cuda-withlinear++fw",
            # fwbw
            # "lstm--vanilla_fwbw++fwbw",
            # "lstm--vanilla++fwbw",
            # "lstm--triton_fused++fwbw",
            "lstm--cuda_fused++fwbw",
            "lstm--cuda++fwbw",
            # "lstm--triton_fused-withlinear++fwbw",
            "lstm--cuda_fused-withlinear++fwbw",
            "lstm--cuda-withlinear++fwbw",
            ## baselines
            "nn.LSTM--pytorch-float32++fw",
            "nn.LSTM--pytorch-float32++fwbw",
            "nn.LSTM--pytorch-float16++fw",
            "nn.LSTM--pytorch-float16++fwbw",
        ],
        warmup=25,
        rep=1000,
        dtype="bfloat16",
    )

    batch_size_run_configs_additional = create_batch_size_configs(
        batch_size_add_benchmark_config, dh_nh_pairs=[(768, 1)]
    )

    batch_size_add_benchmark_fn = get_runnable_benchmark(
        batch_size_run_configs_additional, batch_size_add_benchmark_config
    )

    batch_size_add_benchmark_fn.run(
        save_path=f"{OUTPUT_DIR}/{batch_size_add_benchmark_config.benchmark_name}",
        print_data=True,
    )
    ### =================


if __name__ == "__main__":
    paperplot_experiments()
    # debug_experiments()
    paper_plot_experiments_additional()
