from itertools import product

import pytest
import torch
from flashrnn.tests.utils import model_test, check_forward
from flashrnn.tests.utils import create_inputs, flashrnn


sizes1 = {
    "batch_size": [2, 8, 16, 64],
    "head_dim": [32, 64, 512, 1024],
    "sequence_size": [1, 3],
    "num_heads": [1, 2, 4],
    "backend": ["cuda_fused", "cuda"],
}

sizes2 = {
    "batch_size": [2, 8, 16, 64],
    "head_dim": [32, 64],
    "sequence_size": [1, 3],
    "num_heads": [16, 64, 128],
    "backend": ["cuda_fused", "cuda"],
}

sizes3 = {
    "batch_size": [2],
    "head_dim": [32],
    "sequence_size": [1],
    "num_heads": [64],
    "backend": ["cuda_fused", "cuda"],
}

sizes4 = {
    "batch_size": [8, 32, 64],
    "head_dim": [512],
    "sequence_size": [1024],
    "num_heads": [1],
    "backend": ["cuda_fused", "cuda"],
}

size_combinations = (
    [dict(zip(sizes1.keys(), vals)) for vals in product(*sizes1.values())]
    + [dict(zip(sizes2.keys(), vals)) for vals in product(*sizes2.values())]
    + [dict(zip(sizes3.keys(), vals)) for vals in product(*sizes3.values())]
    + [dict(zip(sizes4.keys(), vals)) for vals in product(*sizes4.values())]
)


class TestLSTM:
    @pytest.mark.parametrize("size_combination", size_combinations)
    def test_sizes_float32(self, size_combination):
        try:
            assert model_test(
                function="lstm",
                dtype=torch.float32,
                tensor_compare_kwargs={"atol": 0.1, "rtol": 0.2},
                **size_combination,
            )
        except ValueError:
            pass

    @pytest.mark.parametrize(
        "size_combination",
        [
            size_combination
            for size_combination in size_combinations
            if size_combination["num_heads"] == 1
        ],
    )
    def test_torch_lstm(self, size_combination):
        B, S, H = (
            size_combination["batch_size"],
            size_combination["sequence_size"],
            size_combination["head_dim"],
        )
        Wx, R, b = create_inputs(
            function="lstm", create_states=False, **size_combination
        )
        x_ = torch.randn([B, S, H], dtype=torch.float32, device=Wx.device)
        lstm = torch.nn.LSTM(
            H,
            H,
            num_layers=1,
            bias=True,
            batch_first=True,
            bidirectional=False,
            device=Wx.device,
        )

        def lstm_functional(x):
            Wx, R, b = create_inputs(
                function="lstm", create_states=False, **size_combination
            )
            with torch.no_grad():
                Wx_ = x @ lstm.weight_ih_l0.transpose(0, 1)
                R_ = lstm.weight_hh_l0.clone().detach()
                b_ = lstm.bias_hh_l0.clone().detach() + lstm.bias_ih_l0.clone().detach()
            Wx = Wx_.reshape(Wx.shape)
            R = R_.reshape(R.shape)
            b = b_.reshape(b.shape)
            return flashrnn(
                Wx, R, b, function="lstm", dtype="float32", backend="vanilla"
            )[0][0, :, :, 0]

        def lstm_torch(x):
            return lstm(x)[0]

        assert check_forward(
            lstm_functional,
            lstm_torch,
            x_,
            show_plot_diff=False,
            atol=0.001,
            rtol=0.002,
        )

    @pytest.mark.parametrize("size_combination", size_combinations)
    def test_sizes_float16(self, size_combination):
        try:
            assert model_test(
                function="lstm",
                dtype=torch.float16,
                tensor_compare_kwargs={"atol": 0.1, "rtol": 0.2},
                **size_combination,
            )
        except ValueError:
            pass

    @pytest.mark.parametrize("size_combination", size_combinations)
    def test_sizes_bfloat16(self, size_combination):
        try:
            assert model_test(
                function="lstm",
                dtype=torch.bfloat16,
                tensor_compare_kwargs={"atol": 0.2, "rtol": 0.4},
                **size_combination,
            )
        except ValueError:
            pass


if __name__ == "__main__":
    model_test(
        function="lstm",
        dtype=torch.bfloat16,
        tensor_compare_kwargs={"atol": 0.05, "rtol": 0.2},
        **size_combinations[36],
    )
