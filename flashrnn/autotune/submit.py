import subprocess
import shlex
import shutil
import argparse
from tqdm import tqdm
import os
from pathlib import Path
import json
import pandas as pd
from dataclasses import asdict
from xlstm.benchmarking.profiling.xlstm_profiling import ProfilingConfig
from itertools import product


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hidden_dim", type=int, required=True)
    parser.add_argument("--batch_dim", type=int, required=True)
    parser.add_argument("--sequence_dim", type=int, required=True)
    parser.add_argument("--version", type=str, required=True)

    args = parser.parse_args()

    return args


def flatten_dict(input_dict):
    keys = input_dict.keys()
    values = [input_dict[key] for key in keys]

    result = []

    for combination in product(*values):
        flat_dict = {key: value for key, value in zip(keys, combination)}
        result.append(flat_dict)

    return result


def get_command(model_config):
    command = "python -m xlstm.models.xlstms.autotune.runner"
    command += f" --hidden_dim {model_config.hidden_dim}"
    command += f" --batch_dim {model_config.batch_dim}"
    command += f" --sequence_dim {model_config.sequence_dim}"
    command += f" --version {model_config.version}"

    return command


def probe_setting(model_config, kernel_setting):
    env = os.environ.copy()
    env.update(kernel_setting)

    command = get_command(model_config)

    process = subprocess.Popen(
        shlex.split(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    stdout, stderr = process.communicate()
    result = {}
    result.update(kernel_setting)
    result.update(asdict(model_config))

    logdir = Path(os.path.dirname(__file__)) / "logs"
    if logdir.exists():
        shutil.rmtree(logdir)

    logdir.mkdir(parents=True, exist_ok=True)
    logfile = "_".join([str(v) for v in result.values()])

    if "Error" in stderr:
        with open(logdir / logfile, "w") as f:
            f.write(stderr)

        for line in stderr.splitlines():
            if "RuntimeError" in line:
                result["error"] = line.replace("RuntimeError: ", "")

    for line in stdout.splitlines():
        if "#FORWARD" in line:
            result["forward"] = line.replace("#FORWARD", "")
        if "#BACKWARD" in line:
            result["backward"] = line.replace("#BACKWARD", "")

    return pd.Series(result)


def generate_kernel_settings(model_config):
    basedir = Path(os.path.dirname(__file__))
    with open(basedir / "tune.settings", "r") as f:
        tune_settings = json.load(f)

    static = {}
    tuneable = {}
    for k in tune_settings:
        if type(tune_settings[k]) is list:
            tuneable[k] = tune_settings[k]
        else:
            static[k] = str(tune_settings[k])

    keys = tuneable.keys()
    values = [tuneable[key] for key in keys]

    settings = []
    for combination in product(*values):
        flat_dict = {key: str(value) for key, value in zip(keys, combination)}
        settings.append(flat_dict)
        settings[-1].update(static)

    with open(basedir / "tune.rules", "r") as f:
        rules = f.readlines()

    valid_settings = []
    for setting in settings:
        eval_setting = setting.copy()
        eval_setting.update(asdict(model_config))
        passed = True
        for rule in rules:
            if not rule:
                continue

            if not eval(rule.format(**eval_setting)):
                passed = False

        if passed:
            valid_settings.append(setting)

    return valid_settings


if __name__ == "__main__":
    args = create_parser()

    model_config = ProfilingConfig(
        hidden_dim=args.hidden_dim,
        batch_dim=args.batch_dim,
        sequence_dim=args.sequence_dim,
        version=args.version,
    )

    settings = generate_kernel_settings(model_config)
    resultdir = Path(os.path.dirname(__file__)) / "results"
    resultdir.mkdir(parents=True, exist_ok=True)

    results = []
    for setting in tqdm(settings):
        result = probe_setting(model_config, setting)
        results.append(result)

    results = pd.DataFrame(results)

    if (resultdir / "result.pkl").exists():
        tmp = pd.read_pickle(resultdir / "result.pkl")
        results = pd.merge(results, tmp, how="outer")

        params = [c for c in results if c not in ["forward", "backward", "error"]]
        results = results.drop_duplicates(params, keep="last").reset_index(drop=True)

    results.to_pickle(resultdir / "result.pkl")
    print(results)
