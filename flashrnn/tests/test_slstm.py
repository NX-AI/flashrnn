from itertools import product

import pytest
import torch
from flashrnn.tests.utils import model_test


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
    "batch_size": [1],
    "head_dim": [32],
    "sequence_size": [1],
    "num_heads": [1],
    "backend": ["cuda_fused", "cuda"],
}


size_combinations = (
    [dict(zip(sizes1.keys(), vals)) for vals in product(*sizes1.values())]
    + [dict(zip(sizes2.keys(), vals)) for vals in product(*sizes2.values())]
    + [dict(zip(sizes3.keys(), vals)) for vals in product(*sizes3.values())]
)


class TestLSTM:
    @pytest.mark.parametrize("size_combination", size_combinations)
    def test_sizes_float32(self, size_combination):
        try:
            assert model_test(
                function="slstm",
                dtype=torch.float32,
                tensor_compare_kwargs={"atol": 0.1, "rtol": 0.2},
                **size_combination,
            )
        except ValueError:
            pass

    @pytest.mark.parametrize("size_combination", size_combinations)
    def test_sizes_float16(self, size_combination):
        try:
            assert model_test(
                function="slstm",
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
                function="slstm",
                dtype=torch.bfloat16,
                tensor_compare_kwargs={"atol": 0.2, "rtol": 0.4},
                **size_combination,
            )
        except ValueError:
            pass


if __name__ == "__main__":
    model_test(
        function="slstm",
        dtype=torch.float32,
        tensor_compare_kwargs={"atol": 0.1, "rtol": 0.2},
        **size_combinations[1],
        # **size_combinations[36],
    )
    import sys

    if "--check-with-plots" in sys.argv:
        from flashrnn.tests.utils import create_inputs, check_backward, check_forward

        kwargs = size_combinations[46]
        from flashrnn import flashrnn

        Wx, states, R, b = create_inputs(
            function="slstm", create_states=True, dtype=torch.float32, **kwargs
        )

        check_forward(
            lambda Wx, R, b, states: flashrnn(
                Wx,
                R,
                b,
                states,
                function="slstm",
                backend="cuda_fused",
                dtype="float32",
            )[0][0],
            lambda Wx, R, b, states: flashrnn(
                Wx,
                R,
                b,
                states,
                function="slstm",
                backend="vanilla",
                dtype="float32",
            )[0][0],
            inputs=(Wx, R, b, states),
            show_plot_diff=True,
            rtol=0.05,
            atol=0.01,
        )

        check_backward(
            lambda Wx, R, b, states: flashrnn(
                Wx,
                R,
                b,
                states,
                function="slstm",
                backend="cuda_fused",
                dtype="float32",
            )[0][0],
            lambda Wx, R, b, states: flashrnn(
                Wx,
                R,
                b,
                states,
                function="slstm",
                backend="vanilla",
                dtype="float32",
            )[0][0],
            inputs=(Wx, R, b, states),
            input_grad_mask=(None, None, None, lambda x: torch.tensor(0.0)),
            show_plot_diff=True,
            rtol=0.05,
            atol=0.01,
        )
