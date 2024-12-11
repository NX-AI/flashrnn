# Copyright 2024 NXAI GmbH
# Korbinian Poeppel
import os
from pathlib import Path

import torch

from ..cuda_init import load

curdir = Path(os.path.split(os.path.os.path.abspath(__file__))[0])


class _GPUInfoCUDA(object):
    mod = None

    @classmethod
    def instance(cls):
        if cls.mod is None:
            if "TORCH_EXTENSIONS_DIR" in os.environ:
                lockfile = os.path.join(
                    os.environ["TORCH_EXTENSIONS_DIR"], "gpu_info", "lock"
                )
                _ = lockfile
                # if os.path.exists(lockfile):
                #     os.remove(lockfile)

            cls.mod = load(
                name="gpu_info2",
                sources=[
                    str(curdir / "gpu_info.cc"),
                    str(curdir / "gpu_info.cu"),
                ],
            )
        return cls.mod


def get_gpu_info(device_id: int) -> dict:
    if device_id >= torch.cuda.device_count():
        return {}
    gpu_info_cuda = _GPUInfoCUDA.instance()
    gpu_info = gpu_info_cuda.GPUInfo()

    return gpu_info.gpu_info(device_id)
