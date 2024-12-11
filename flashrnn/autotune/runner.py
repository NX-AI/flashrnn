import argparse
from xlstm.benchmarking.profiling.xlstm_profiling import get_inputs
from xlstm.benchmarking.profiling.xlstm_profiling import run_single
from xlstm.benchmarking.profiling.xlstm_profiling import prepare_models
from xlstm.benchmarking.profiling.xlstm_profiling import ProfilingConfig


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hidden_dim", type=int, required=True)
    parser.add_argument("--batch_dim", type=int, required=True)
    parser.add_argument("--sequence_dim", type=int, required=True)
    parser.add_argument("--version", type=str, required=True)

    args = parser.parse_args()

    return args


def measure(config):
    model = prepare_models(config, which="fused")["fused"]
    input = get_inputs(model, config)

    return run_single(model, input)


if __name__ == "__main__":
    args = create_parser()

    config = ProfilingConfig(
        hidden_dim=args.hidden_dim,
        batch_dim=args.batch_dim,
        sequence_dim=args.sequence_dim,
        version=args.version,
    )

    result = measure(config)

    fw_time = result[result["name"] == "forward_pass"]["cuda_time"].values
    bw_time = result[result["name"] == "backward_pass"]["cuda_time"].values

    if len(fw_time) == 1:
        print(f"#FORWARD{fw_time[0]}")
    if len(bw_time) == 1:
        print(f"#BACKWARD{bw_time[0]}")
