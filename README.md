# CUDA Kernel Tuning Agent

I built this project to learn how GPU matrix multiplication is optimized, and to see whether an
LLM agent could help search for faster kernel configurations.

The agent picks a kernel configuration, compiles it with `nvcc`, benchmarks it against cuBLAS,
checks that the output is correct, and logs the result. It then looks at everything it has tried
so far and decides what to test next.

## How it works

```
propose -> compile (nvcc) -> benchmark + correctness check -> log to SQLite -> analyze
   ^                                                                            |
   +------------------------------ next iteration ------------------------------+
```

### Kernels (`kernels/gemm.cu`)

All three kernels compute single-precision matrix multiplication (C = A x B) in row-major order.

1. **Naive:** each thread computes one output element and reads directly from global memory.
2. **Shared-memory tiling:** each block loads tiles of A and B into shared memory, so every
   value read from global memory is reused across the whole tile.
3. **Tiling with thread coarsening:** each thread computes several output rows instead of one,
   which reuses values loaded from shared memory and keeps more work in registers.

Each kernel is timed with CUDA events over several runs after a warm-up, and compared with
cuBLAS for both speed and correctness. Since cuBLAS is column-major, the baseline computes
C^T = B^T x A^T, which gives the same result as the row-major product.

### Benchmark harness (`bench/run_benchmark.py`)

Before compiling, the harness checks each configuration against hardware limits: at most 1024
threads per block and 48 KB of shared memory per block. It then compiles the configuration,
runs it, and records:

- runtime (ms) and GFLOPS
- percentage of cuBLAS performance
- compile time
- maximum error compared with cuBLAS

### Agent (`agent/graph.py`)

The agent is a LangGraph loop with three steps: propose, evaluate, and analyze.

- If `OPENAI_API_KEY` is set, an LLM reads the results so far and suggests the next
  configuration, along with a short reason.
- If the key is not set, or the suggestion is invalid or already tested, it falls back to
  working through a fixed search space, so the loop always finishes.

Every run is saved to SQLite (`agent/store.py`), so different runs can be compared later.

### Profiling (`bench/profile.py`)

Collects Nsight Compute metrics for a configuration (compute throughput, memory throughput, and
achieved occupancy) to help explain why one kernel is faster than another.

## Project structure

```
kernels/gemm.cu           CUDA kernels, cuBLAS baseline, and correctness check
bench/run_benchmark.py    validates, compiles, and runs a configuration
bench/profile.py          Nsight Compute metrics
agent/graph.py            LangGraph tuning loop
agent/store.py            SQLite experiment log
scripts/plot_results.py   GFLOPS chart compared with cuBLAS
Makefile                  build shortcuts
```

## Running it

I ran this on Google Colab with an NVIDIA T4 GPU (Runtime > Change runtime type > T4 GPU).

```bash
git clone https://github.com/charmikadi/cuda-kernel-tuning-agent.git
cd cuda-kernel-tuning-agent
pip install -r requirements.txt

make all                           # build the three kernels
python -m bench.run_benchmark      # benchmark them against cuBLAS

export OPENAI_API_KEY=your_key     # optional, enables the LLM proposer
python -m agent.graph --iters 12   # run the tuning agent
python -m scripts.plot_results     # saves results/gflops.png
python -m bench.profile            # optional, needs Nsight Compute
```

The default architecture is `sm_75` (T4). For other GPUs, set `CUDA_ARCH` and pass `ARCH` to
make, for example `sm_80` for an A100 or `sm_86` for an RTX 30-series card:

```bash
export CUDA_ARCH=sm_80
make all ARCH=sm_80
```

## Tech stack

CUDA, C++, cuBLAS, Nsight Compute, Python, LangGraph, OpenAI API, SQLite, Matplotlib
