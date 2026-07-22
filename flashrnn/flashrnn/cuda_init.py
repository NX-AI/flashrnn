# Copyright 2024 NXAI GmbH
# Korbinian Poeppel

import os
from typing import Sequence, Union
import logging
import functools
import subprocess
import sys
from packaging.version import Version
import re

import time
import random

import torch
from torch.utils.cpp_extension import load as _load
from torch.utils.cpp_extension import  _find_cuda_home 


# print("INCLUDE:", torch.utils.cpp_extension.include_paths(cuda=True))
# print("C++ compat", torch.utils.cpp_extension.check_compiler_abi_compatibility("g++"))
# print("C compat", torch.utils.cpp_extension.check_compiler_abi_compatibility("gcc"))

LOGGER = logging.getLogger(__name__)
CUDA_HOME = _find_cuda_home()
IS_WINDOWS = sys.platform == 'win32'
IS_MACOS = sys.platform.startswith('darwin')
SUBPROCESS_DECODE_ARGS = ('oem',) if IS_WINDOWS else ()

@functools.cache
def get_cuda_version() -> Version | None:
    try:
        nvcc = os.path.join(CUDA_HOME, 'bin', 'nvcc.exe' if IS_WINDOWS else 'nvcc')
        cuda_version_str = subprocess.check_output([nvcc, '--version']).strip().decode(*SUBPROCESS_DECODE_ARGS)
        cuda_version = re.search(r'release (\d+[.]\d+)', cuda_version_str)

        if cuda_version is None:
            return
        cuda_str_version = cuda_version.group(1)
        cuda_version = Version(cuda_str_version)
        return cuda_version
    except (RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as e:
        # raise RuntimeError(
        #         "Could not determine CUDA version."
        #     ) from e
        return 


def defines_to_cflags(
    defines=Union[dict[str, Union[int, str]], Sequence[tuple[str, Union[str, int]]]],
):
    cflags = []
    LOGGER.info("Compiling definitions: ", defines)
    if isinstance(defines, dict):
        defines = defines.items()
    for key, val in defines:
        cflags.append(f"-D{key}={str(val)}")
    return cflags


curdir = os.path.dirname(__file__)

if torch.cuda.is_available():
    from packaging import version

    if version.parse(torch.__version__) >= version.parse("2.6.0"):
        os.environ["CUDA_LIB"] = os.path.join(
            os.path.split(torch.utils.cpp_extension.include_paths(device_type="cuda")[-1])[0], "lib"
        )
    else:
        os.environ["CUDA_LIB"] = os.path.join(
            os.path.split(torch.utils.cpp_extension.include_paths(cuda=True)[-1])[0], "lib"
        )


EXTRA_INCLUDE_PATHS = () + (
    tuple(os.environ["FLASHRNN_EXTRA_INCLUDE_PATHS"].split(":")) if "FLASHRNN_EXTRA_INCLUDE_PATHS" in os.environ else ()
)
if "CONDA_PREFIX" in os.environ:
    # This enforces adding the correct include directory from the CUDA installation via torch. If you use the system
    # installation, you might have to add the cflags yourself.
    from pathlib import Path
    from packaging import version
    import glob

    if version.parse(torch.__version__) >= version.parse("2.6.0"):
        matching_dirs = glob.glob(f"{os.environ['CONDA_PREFIX']}/targets/**", recursive=True)
        EXTRA_INCLUDE_PATHS = (
            EXTRA_INCLUDE_PATHS
            + tuple(map(str, (Path(os.environ["CONDA_PREFIX"]) / "targets").glob("**/include/")))[:1]
        )


def load(*, name, sources, extra_cflags=(), extra_cuda_cflags=(), **kwargs):
    suffix = ""
    for flag in extra_cflags:
        pref = [st[0] for st in flag[2:].split("=")[0].split("_")]
        if len(pref) > 1:
            pref = pref[1:]
        suffix += "".join(pref)
        value = flag[2:].split("=")[1].replace("-", "m").replace(".", "d")
        value_map = {
            "float": "f",
            "__half": "h",
            "__nv_bfloat16": "b",
            "true": "1",
            "false": "0",
        }
        if value in value_map:
            value = value_map[value]
        suffix += value
    if suffix:
        suffix = "_" + suffix
    suffix = suffix[:64]

    extra_cflags = list(extra_cflags) + [
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
    ]
    for eip in EXTRA_INCLUDE_PATHS:
        extra_cflags.append("-isystem")
        extra_cflags.append(eip)

    conda_prefix = os.getenv("CONDA_PREFIX")
    if conda_prefix:
        conda_prefix_include = os.path.join(conda_prefix, "include")
        extra_cflags.append(f"-I{conda_prefix_include}")
        extra_cuda_cflags = list(extra_cuda_cflags) + [f"-I{conda_prefix_include}"]

    # Windows: MSVC link.exe needs /LIBPATH: and *.lib, not the Unix -L / -l form.
    if IS_WINDOWS:
        _win_cuda_libdir = os.path.join(CUDA_HOME, "lib", "x64")
        _extra_ldflags = [f"/LIBPATH:{_win_cuda_libdir}", "cublas.lib"]
    else:
        _extra_ldflags = [f"-L{os.environ['CUDA_LIB']}", "-lcublas"]

    myargs = {
        "verbose": True,
        "with_cuda": True,
        "extra_ldflags": _extra_ldflags,
        "extra_cflags": [*extra_cflags],
        "extra_cuda_cflags": [
            # "-gencode",
            # "arch=compute_70,code=compute_70",
            # "-dbg=1",
            '-Xptxas="-v"',
            "-gencode",
            "arch=compute_80,code=compute_80",
            "-res-usage",
            "--use_fast_math",
            "-O3",
            # Windows nvcc rejects the joined token "-Xptxas -O3"; pass as two args.
            "-Xptxas",
            "-O3",
            "--extra-device-vectorization",
            *extra_cflags,
            *extra_cuda_cflags,
        ],
    }

    cuda_version = get_cuda_version()
    if cuda_version is not None and cuda_version >= Version("12.8"):
        myargs['extra_cuda_cflags'].append("-static-global-template-stub=false")

    LOGGER.info("Kernel compilation arguments", myargs)
    myargs.update(**kwargs)
    # add random waiting time to minimize deadlocks because of badly managed multicompile of pytorch ext
    time.sleep(random.random() * 10)
    LOGGER.info(f"Before compilation and loading of {name}.")
    mod = _load(name + suffix, sources, **myargs)
    LOGGER.info(f"After compilation and loading of {name}.")
    return mod
