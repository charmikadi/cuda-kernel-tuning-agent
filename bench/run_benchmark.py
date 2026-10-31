"""Compile one GEMM configuration with nvcc, run it, and return the measurements."""
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "kernels" / "gemm.cu"
BUILD = ROOT / "build"
ARCH = os.environ.get("CUDA_ARCH", "sm_75")  # sm_75 = T4 (Google Colab). Use sm_80 for A100.

KERNEL_NAMES = {0: "naive", 1: "tiled", 2: "coarsened"}


def validate(config: dict) -> str | None:
    """Return an error message if the configuration cannot run, otherwise None."""
    kernel, tile, coarsen = config["kernel"], config.get("tile", 32), config.get("coarsen", 1)
    if kernel not in KERNEL_NAMES:
        return f"unknown kernel {kernel}"
    if kernel == 0:
        return None
    if tile not in (8, 16, 32, 64):
        return "tile must be 8, 16, 32, or 64"
    shared_bytes = 2 * tile * tile * 4
    if shared_bytes > 48 * 1024:
        return f"needs {shared_bytes} bytes of shared memory (limit 48 KB)"
    threads = tile * tile if kernel == 1 else tile * (tile // max(coarsen, 1))
    if kernel == 2 and (coarsen < 1 or tile % coarsen != 0):
        return "tile must be divisible by coarsen"
    if threads > 1024:
        return f"{threads} threads per block (limit 1024)"
    return None


def run(config: dict, size: int = 2048, iters: int = 20) -> dict:
    """Compile and benchmark a configuration. Always returns a result dict."""
    error = validate(config)
    if error:
        return {**config, "status": "invalid", "error": error}

    kernel, tile, coarsen = config["kernel"], config.get("tile", 32), config.get("coarsen", 1)
    BUILD.mkdir(exist_ok=True)
    binary = BUILD / f"gemm_k{kernel}_t{tile}_c{coarsen}"
    cmd = ["nvcc", "-O3", f"-arch={ARCH}", f"-DKERNEL={kernel}", f"-DTILE={tile}",
           f"-DCOARSEN={coarsen}", str(SRC), "-lcublas", "-o", str(binary)]

    t0 = time.perf_counter()
    compiled = subprocess.run(cmd, capture_output=True, text=True)
    compile_s = time.perf_counter() - t0
    if compiled.returncode != 0:
        return {**config, "status": "compile_error", "compile_s": compile_s,
                "error": compiled.stderr[-2000:]}

    ran = subprocess.run([str(binary), str(size), str(size), str(size), str(iters)],
                         capture_output=True, text=True, timeout=300)
    if ran.returncode != 0:
        return {**config, "status": "runtime_error", "compile_s": compile_s,
                "error": ran.stderr[-2000:]}

    result = json.loads(ran.stdout.strip().splitlines()[-1])
    result.update(status="ok" if result["correct"] else "incorrect", compile_s=round(compile_s, 2))
    return result


if __name__ == "__main__":
    for cfg in ({"kernel": 0}, {"kernel": 1, "tile": 32}, {"kernel": 2, "tile": 64, "coarsen": 8}):
        print(run(cfg))
