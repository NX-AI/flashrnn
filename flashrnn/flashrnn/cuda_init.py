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
# ROCm builds of torch expose the CUDA API through HIP: torch.cuda works and
# .cu sources are hipified transparently, but there is no nvcc / CUDA_HOME and
# the toolchain is hipcc, so the load path below swaps compiler flags and links
# hipBLAS instead of cuBLAS.
IS_HIP = torch.version.hip is not None

@functools.cache
def get_cuda_version() -> Version | None:
    # On ROCm (and CPU-only) machines there is no CUDA toolkit, so CUDA_HOME is
    # None; os.path.join(None, ...) would raise TypeError, which the except
    # clause below did not catch. Bail out early instead of crashing.
    if CUDA_HOME is None:
        return None
    try:
        nvcc = os.path.join(CUDA_HOME, 'bin', 'nvcc.exe' if IS_WINDOWS else 'nvcc')
        cuda_version_str = subprocess.check_output([nvcc, '--version']).strip().decode(*SUBPROCESS_DECODE_ARGS)
        cuda_version = re.search(r'release (\d+[.]\d+)', cuda_version_str)

        if cuda_version is None:
            return
        cuda_str_version = cuda_version.group(1)
        cuda_version = Version(cuda_str_version)
        return cuda_version
    except (RuntimeError, FileNotFoundError, subprocess.CalledProcessError, TypeError) as e:
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

    if IS_HIP:
        from torch.utils.cpp_extension import ROCM_HOME

        os.environ["CUDA_LIB"] = os.path.join(ROCM_HOME or "/opt/rocm", "lib")
    elif version.parse(torch.__version__) >= version.parse("2.6.0"):
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


def _hipify_sources(sources):
    """Translate the FlashRNN CUDA sources — and the headers they include — to
    HIP.

    torch's JIT builder only hipifies the files passed as ``sources``, not the
    headers they pull in (blas.h, inline_ops*.cuh, support.h, ...), so those
    would keep their cuBLAS / ``__nv_bfloat16`` spellings while the hipified .cu
    bodies use the HIP ones. Instead copy the needed subtrees into a cache dir,
    hipify everything there, and compile from that copy; the repo sources are
    never touched.
    """
    import shutil
    from torch.utils.hipify import hipify_python
    from torch.utils.file_baton import FileBaton

    src_root = os.path.abspath(curdir)
    out_root = os.environ.get(
        "FLASHRNN_HIP_SRC_DIR",
        os.path.join(
            os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
            "flashrnn",
            "hip_src",
        ),
    )

    def _map_source(source):
        rel = os.path.relpath(os.path.abspath(source), src_root)
        hip_rel = hipify_python.get_hip_file_path(rel, is_pytorch_extension=True)
        hip_path = os.path.join(out_root, hip_rel)
        if not os.path.exists(hip_path):
            hip_path = os.path.join(out_root, rel)
        # The pybind glue (.cc) references the HIP fp16/bf16 types for its dtype
        # dispatch (support.h). Those headers only compile under clang, so route
        # the glue through hipcc by giving it a .hip extension (torch selects the
        # compiler by suffix); a plain-C++ host compile would fail.
        if hip_path.endswith((".cc", ".cpp")):
            hip_path = os.path.splitext(hip_path)[0] + "_glue.hip"
        return hip_path

    def _produce():
        # Subdirs referenced by the sources (alternating/ or fused/), plus util/.
        subdirs = {os.path.relpath(os.path.dirname(os.path.abspath(s)), src_root) for s in sources}
        subdirs.add("util")
        for sub in sorted(subdirs):
            src_sub = os.path.join(src_root, sub)
            if os.path.isdir(src_sub):
                shutil.copytree(src_sub, os.path.join(out_root, sub), dirs_exist_ok=True)
        hipify_python.hipify(
            project_directory=out_root,
            output_directory=out_root,
            includes=[os.path.join(out_root, "*")],
            is_pytorch_extension=True,
            show_detailed=False,
        )
        # hipify writes renamed copies (foo.cu -> foo.hip, blas.h -> blas_hip.h)
        # and leaves the untranslated originals; overwrite the originals with the
        # hipified text so stale relative includes still resolve to HIP.
        for dirpath, _, filenames in os.walk(out_root):
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                rel = os.path.relpath(path, out_root)
                hip_rel = hipify_python.get_hip_file_path(rel, is_pytorch_extension=True)
                hip_path = os.path.join(out_root, hip_rel)
                if hip_path != path and os.path.exists(hip_path):
                    shutil.copyfile(hip_path, path)

        # Residual fixups hipify's substitution map does not cover.
        import re

        _dead_includes = re.compile(
            r'^\s*#\s*include\s*[<"](?:cuda|cuda_runtime_api|cuda_device_runtime_api)\.h[>"]\s*$',
            re.MULTILINE,
        )
        _text_exts = (".cu", ".cuh", ".cc", ".cpp", ".c", ".h", ".hpp", ".hip")
        for dirpath, _, filenames in os.walk(out_root):
            if "__pycache__" in dirpath:
                continue
            for filename in filenames:
                if not filename.endswith(_text_exts):
                    continue  # skip .pyc and other binaries copied alongside sources
                path = os.path.join(dirpath, filename)
                with open(path, "r") as fh:
                    text = fh.read()
                new_text = _dead_includes.sub("// [hip] removed CUDA-only include", text)
                # hipify rewrites &cublasHgemm -> &hipblasHgemm, whose hipblasHalf
                # signature is incompatible with the __half-typed wrapper; use the
                # local cublasHgemm2 wrapper (matches the strided path) instead.
                new_text = re.sub(r"&\s*hipblasHgemm\b", "&cublasHgemm2", new_text)
                # bf16 blas support is gated on CUDART_VERSION, which HIP lacks;
                # enable the same block on ROCm.
                new_text = new_text.replace(
                    "CUDART_VERSION >= 11020",
                    "(CUDART_VERSION >= 11020 || defined(__HIP_PLATFORM_AMD__))",
                )
                if new_text != text:
                    with open(path, "w") as fh:
                        fh.write(new_text)

        for source in sources:
            if source.endswith((".cc", ".cpp")):
                rel = os.path.relpath(os.path.abspath(source), src_root)
                hip_rel = hipify_python.get_hip_file_path(rel, is_pytorch_extension=True)
                orig = os.path.join(out_root, hip_rel)
                if not os.path.exists(orig):
                    orig = os.path.join(out_root, rel)
                shutil.copyfile(orig, _map_source(source))

    # Serialize the shared cache tree across concurrent workers (e.g. several
    # dataloader / distributed processes initializing the backend at once): the
    # first to arrive builds it, the rest wait for that build to finish.
    os.makedirs(os.path.dirname(out_root), exist_ok=True)
    baton = FileBaton(out_root.rstrip("/") + ".lock")
    if baton.try_acquire():
        try:
            _produce()
        finally:
            baton.release()
    else:
        baton.wait()

    return [_map_source(s) for s in sources], out_root


def _remap_hip_includes(flags, out_root):
    """Point any `-include <repo path>` at its hipified copy in the cache tree.

    The fused backend force includes fused/{function}_fused_pointwise.cuh from
    the original source tree. On ROCm that header must be the hipified version,
    otherwise its cuBLAS/bf16 spellings reach the HIP compiler untranslated.
    """
    src_root = os.path.abspath(curdir)
    flags = list(flags)
    out = []
    i = 0
    while i < len(flags):
        out.append(flags[i])
        if flags[i] == "-include" and i + 1 < len(flags):
            inc = os.path.abspath(flags[i + 1])
            if inc.startswith(src_root + os.sep):
                cand = os.path.join(out_root, os.path.relpath(inc, src_root))
                out.append(cand if os.path.exists(cand) else flags[i + 1])
            else:
                out.append(flags[i + 1])
            i += 2
            continue
        i += 1
    return out


def load(*, name, sources, extra_cflags=(), extra_cuda_cflags=(), **kwargs):
    if IS_HIP:
        sources, hip_out_root = _hipify_sources(sources)
        extra_cuda_cflags = _remap_hip_includes(extra_cuda_cflags, hip_out_root)
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
            "__hip_bfloat16": "b",
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

    if IS_HIP:
        # hipcc rejects nvcc-only flags (-Xptxas, -gencode, -res-usage, ...);
        # cuBLAS calls hipify to hipBLAS, so link hipblas. The compat shim is
        # force-included ONLY on the device pass: it pulls in hip_bf16.h /
        # hip_fp16.h, which rely on clang builtins and do not compile under the
        # g++ host compiler used for the pybind glue.
        compat_header = os.path.join(curdir, "util", "hip_compat.h")
        myargs = {
            "verbose": True,
            "with_cuda": True,
            "extra_ldflags": [f"-L{os.environ['CUDA_LIB']}", "-lhipblas"],
            "extra_cflags": [*extra_cflags],
            "extra_cuda_cflags": [
                "-O3",
                "-include",
                compat_header,
                *extra_cflags,
                *extra_cuda_cflags,
            ],
        }
    else:
        myargs = {
            "verbose": True,
            "with_cuda": True,
            "extra_ldflags": [f"-L{os.environ['CUDA_LIB']}", "-lcublas"],
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
                "-Xptxas -O3",
                "--extra-device-vectorization",
                *extra_cflags,
                *extra_cuda_cflags,
            ],
        }

    # nvcc-only flag: never add it on HIP builds, even if a CUDA toolkit also
    # happens to be discoverable on the ROCm host (hipcc would reject it).
    if not IS_HIP:
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
