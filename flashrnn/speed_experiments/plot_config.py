import matplotlib as mpl
from pathlib import Path

FONTSIZE = 12
SMALL_OFFSET = 1
FONTSIZE_SMALL = FONTSIZE - SMALL_OFFSET
FONTSIZE_TICKS = 14

FIGSIZE = (2 * 12 * 1 / 2.54, 2 * 8 * 1 / 2.54)
FIGSIZE_2COL = (4 * 0.7 * 12 * 1 / 2.54, 2 * 0.7 * 8 * 1 / 2.54)

GRIDSPEC_KWARGS = {"wspace": 0.115, "hspace": 0}

# SLOW_COLS = [
#     "lstm--vanilla_fwbw++fw",
#     "lstm--vanilla_fwbw++fwbw",
#     "slstm--vanilla_fwbw++fw",
#     "slstm--vanilla_fwbw++fwbw",
# ]

save_path = Path(".") / "paperplots"


kernel_colors = {
    "vanilla": mpl.colormaps["tab10"].colors[0],
    "triton_fused": mpl.colormaps["tab10"].colors[1],
    "cuda_fused": mpl.colormaps["tab10"].colors[2],
    "cuda": mpl.colormaps["tab10"].colors[3],
    "cuda_fused_wl": mpl.colormaps["tab10"].colors[5],
    "cuda_wl": mpl.colormaps["tab10"].colors[6],
    "attention": mpl.colormaps["tab10"].colors[4],
    "lstm_pt_bf16": mpl.colormaps["tab10"].colors[7],
    "lstm_pt_fp32": mpl.colormaps["tab10"].colors[8],
    "lstm_pt_fp16": mpl.colormaps["tab10"].colors[8],
    "lstm_haste_fp32": mpl.colormaps["tab10"].colors[9],
}

kernel_labels = {
    "vanilla": "Vanilla PyTorch",
    "triton_fused": "Triton fused",
    "cuda_fused": "CUDA fused",
    "cuda": "CUDA alternating",
    "cuda_fused_wl": "CUDA fused w/ Linear",
    "cuda_wl": "CUDA alternating w/ Linear",
    "attention": "FlashAttention2",
    "lstm_pt_bf16": "PT nn.LSTM bf16",
    "lstm_pt_fp32": "PT nn.LSTM fp32",
    "lstm_pt_fp16": "PT nn.LSTM",
    "lstm_haste_fp32": "Haste LSTM fp32",
}

style_dict = {
    "lstm--vanilla_fwbw++fw": {
        "color": kernel_colors["vanilla"],
        "label": kernel_labels["vanilla"],
    },
    "lstm--triton_fused++fw": {
        "color": kernel_colors["triton_fused"],
        "label": kernel_labels["triton_fused"],
    },
    "lstm--cuda_fused++fw": {
        "color": kernel_colors["cuda_fused"],
        "label": kernel_labels["cuda_fused"],
    },
    "lstm--cuda_fused-withlinear++fw": {
        "color": kernel_colors["cuda_fused_wl"],
        "label": kernel_labels["cuda_fused_wl"],
    },
    "lstm--cuda++fw": {"color": kernel_colors["cuda"], "label": kernel_labels["cuda"]},
    "lstm--cuda-withlinear++fw": {
        "color": kernel_colors["cuda_wl"],
        "label": kernel_labels["cuda_wl"],
    },
    "lstm--vanilla_fwbw++fwbw": {
        "color": kernel_colors["vanilla"],
        "label": kernel_labels["vanilla"],
    },
    "lstm--triton_fused++fwbw": {
        "color": kernel_colors["triton_fused"],
        "label": kernel_labels["triton_fused"],
    },
    "lstm--cuda_fused++fwbw": {
        "color": kernel_colors["cuda_fused"],
        "label": kernel_labels["cuda_fused"],
    },
    "lstm--cuda_fused-withlinear++fwbw": {
        "color": kernel_colors["cuda_fused_wl"],
        "label": kernel_labels["cuda_fused_wl"],
    },
    "lstm--cuda++fwbw": {
        "color": kernel_colors["cuda"],
        "label": kernel_labels["cuda"],
    },
    "lstm--cuda-withlinear++fwbw": {
        "color": kernel_colors["cuda_wl"],
        "label": kernel_labels["cuda_wl"],
    },
    "slstm--vanilla_fwbw++fw": {
        "color": kernel_colors["vanilla"],
        "label": kernel_labels["vanilla"],
    },
    "slstm--triton_fused++fw": {
        "color": kernel_colors["triton_fused"],
        "label": kernel_labels["triton_fused"],
    },
    "slstm--cuda_fused++fw": {
        "color": kernel_colors["cuda_fused"],
        "label": kernel_labels["cuda_fused"],
    },
    "slstm--cuda++fw": {"color": kernel_colors["cuda"], "label": kernel_labels["cuda"]},
    "slstm--vanilla_fwbw++fwbw": {
        "color": kernel_colors["vanilla"],
        "label": kernel_labels["vanilla"],
    },
    "slstm--triton_fused++fwbw": {
        "color": kernel_colors["triton_fused"],
        "label": kernel_labels["triton_fused"],
    },
    "slstm--cuda_fused++fwbw": {
        "color": kernel_colors["cuda_fused"],
        "label": kernel_labels["cuda_fused"],
    },
    "slstm--cuda++fwbw": {
        "color": kernel_colors["cuda"],
        "label": kernel_labels["cuda"],
    },
    "attention_causal--fa2++fw": {
        "color": kernel_colors["attention"],
        "label": kernel_labels["attention"],
    },
    "attention_causal--fa2++fwbw": {
        "color": kernel_colors["attention"],
        "label": kernel_labels["attention"],
    },
    "nn.LSTM--pytorch-bfloat16++fw": {
        "color": kernel_colors["lstm_pt_bf16"],
        "label": kernel_labels["lstm_pt_bf16"],
    },
    "nn.LSTM--pytorch-bfloat16++fwbw": {
        "color": kernel_colors["lstm_pt_bf16"],
        "label": kernel_labels["lstm_pt_bf16"],
    },
    "nn.LSTM--pytorch-float32++fw": {
        "color": kernel_colors["lstm_pt_fp32"],
        "label": kernel_labels["lstm_pt_fp32"],
    },
    "nn.LSTM--pytorch-float32++fwbw": {
        "color": kernel_colors["lstm_pt_fp32"],
        "label": kernel_labels["lstm_pt_fp32"],
    },
    "nn.LSTM--pytorch-float16++fw": {
        "color": kernel_colors["lstm_pt_fp16"],
        "label": kernel_labels["lstm_pt_fp16"],
    },
    "nn.LSTM--pytorch-float16++fwbw": {
        "color": kernel_colors["lstm_pt_fp16"],
        "label": kernel_labels["lstm_pt_fp16"],
    },
    "haste.LSTM--pytorch-float32++fw": {
        "color": kernel_colors["lstm_haste_fp32"],
        "label": kernel_labels["lstm_haste_fp32"],
    },
    "haste.LSTM--pytorch-float32++fwbw": {
        "color": kernel_colors["lstm_haste_fp32"],
        "label": kernel_labels["lstm_haste_fp32"],
    },
}

col_order_lstm_fw = [
    "lstm--triton_fused++fw",
    "lstm--cuda_fused++fw",
    "lstm--cuda++fw",
    "lstm--vanilla_fwbw++fw",
]
col_order_lstm_wbl_fw = col_order_lstm_fw + [
    "nn.LSTM--pytorch-float32++fw",
    "nn.LSTM--pytorch-bfloat16++fw",
    "nn.LSTM--pytorch-float16++fw",
]

col_order_lstm_fwbw = [
    "lstm--triton_fused++fwbw",
    "lstm--cuda_fused++fwbw",
    "lstm--cuda++fwbw",
    "lstm--vanilla_fwbw++fwbw",
]
col_order_lstm_wbl_fwbw = col_order_lstm_fwbw + [
    "nn.LSTM--pytorch-float32++fwbw",
    "nn.LSTM--pytorch-bfloat16++fwbw",
    "nn.LSTM--pytorch-float16++fwbw",
]

col_order_slstm_fw = [
    "slstm--triton_fused++fw",
    "slstm--cuda_fused++fw",
    "slstm--cuda++fw",
    "slstm--vanilla_fwbw++fw",
]

col_order_slstm_fwbw = [
    "slstm--triton_fused++fwbw",
    "slstm--cuda_fused++fwbw",
    "slstm--cuda++fwbw",
    "slstm--vanilla_fwbw++fwbw",
]
