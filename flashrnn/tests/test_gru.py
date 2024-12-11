from itertools import product

import pytest
import torch
from flashrnn.tests.utils import model_test

# cuda_fused backend is not yet supported as num_gates_r does not divide tensor core sizes, padding needed
#
sizes1 = {
    "batch_size": [2, 8, 16, 64],
    "head_dim": [32, 64, 512, 1024],
    "sequence_size": [1, 3],
    "num_heads": [1, 2, 4],
    "backend": ["cuda"],
}

sizes2 = {
    "batch_size": [2, 8, 16, 64],
    "head_dim": [32, 64],
    "sequence_size": [1, 3],
    "num_heads": [16, 64, 128],
    "backend": ["cuda"],
}

sizes3 = {
    "batch_size": [2],
    "head_dim": [32],
    "sequence_size": [1],
    "num_heads": [64],
    "backend": ["cuda"],
}

sizes4 = {
    "batch_size": [8, 32, 64],
    "head_dim": [512],
    "sequence_size": [1024],
    "num_heads": [1],
    "backend": ["cuda"],
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
                function="gru",
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
                function="gru",
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
                function="gru",
                dtype=torch.bfloat16,
                tensor_compare_kwargs={"atol": 0.2, "rtol": 0.4},
                **size_combination,
            )
        except ValueError:
            pass


if __name__ == "__main__":
    comb = 0
    print(size_combinations[comb])
    model_test(
        function="gru",
        dtype=torch.float16,
        tensor_compare_kwargs={"atol": 0.05, "rtol": 0.2},
        show_plot_diff=True,
        **size_combinations[comb],
    )
