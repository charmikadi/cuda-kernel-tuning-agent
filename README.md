# CUDA Kernel Tuning Agent

An AI agent that experiments with CUDA matrix multiplication (SGEMM) kernels and uses benchmark results to guide its own optimization decisions.

The agent proposes kernel configurations, compiles them with `nvcc`, benchmarks them against NVIDIA cuBLAS, validates correctness, and logs each experiment to SQLite. An LLM analyzes previous results and suggests what configuration to test next.

The goal was to explore GPU performance optimization while building an agent that interacts with real compilation, benchmarking, and profiling workflows.

## Results

Benchmarked on an NVIDIA Tesla T4 GPU (Google Colab) using 2048 × 2048 single-precision matrices (FP32).

| Kernel | Configuration | GFLOPS | Speedup vs. Naive | % of cuBLAS |
|---|---|---:|---:|---:|
| Naive | 16 × 16 block | 423 | 1.0× | 8.9% |
| Tiled | TILE=32 | 799 | 1.9× | 16.1% |
| Coarsened | TILE=32, COARSEN=4 | 1,367 | 3.2× | 29.9% |
| **Best Coarsened** | **TILE=32, COARSEN=16** | **1,749** | **4.1×** | **38.3%** |

The tuning agent evaluated 12 configurations, with the best-performing kernel reaching 1,749 GFLOPS, a 4.1× improvement over the naive implementation. All tested configurations passed correctness validation against cuBLAS.

![GFLOPS comparison](results/gflops.png)

### Key Observations

- **Shared-memory tiling:** Improved performance by approximately 1.9× by reducing redundant global memory accesses through data reuse.
- **Thread coarsening:** Further improved performance by having each thread compute multiple output elements, increasing data reuse and work per thread.
- **Configuration tuning:** Increasing the coarsening factor from 4 to 16 improved performance from 1,367 to 1,749 GFLOPS.

## How It Works

The agent follows an iterative optimization loop:

```text
propose → compile (nvcc) → benchmark + correctness check
   ↑                              ↓
   └──── analyze ← log to SQLite ─┘
```

- **Propose:** LangGraph coordinates the optimization loop. An LLM reviews previous experiments and suggests a new configuration with a short explanation.
- **Evaluate:** The benchmark harness validates hardware constraints, compiles the kernel, measures execution time using CUDA events, and compares results against cuBLAS.
- **Analyze:** Results are stored in SQLite and used to guide subsequent proposals.

If `OPENAI_API_KEY` is not set, or an LLM proposal is invalid or has already been tested, the agent falls back to systematic configuration search.

## CUDA Kernels

Implemented in `kernels/gemm.cu`. All three kernels compute single-precision matrix multiplication, C = A × B.

1. **Naive:** Each thread computes one output element using global memory.
2. **Shared-memory tiling:** Threads cooperatively load matrix tiles into shared memory to reduce redundant global memory accesses.
3. **Tiling + thread coarsening:** Each thread computes multiple output elements, increasing data reuse and reducing redundant work.

cuBLAS uses column-major storage, so the benchmark computes Cᵀ = Bᵀ × Aᵀ to get the equivalent row-major result.

## Benchmarking & Profiling

The benchmark harness (`bench/run_benchmark.py`) validates hardware constraints before compiling and executing each configuration.

It records:
- Kernel execution time and GFLOPS.
- Percentage of cuBLAS performance.
- Compilation time.
- Maximum numerical error and correctness status.

The profiling script (`bench/profile.py`) uses NVIDIA Nsight Compute to collect compute throughput, memory throughput, and achieved occupancy to help identify performance bottlenecks.

## Project Structure

```text
cuda-kernel-tuning-agent/
├── kernels/
│   └── gemm.cu              # CUDA kernel implementations
├── bench/
│   ├── run_benchmark.py     # Benchmarking and correctness
│   └── profile.py           # Nsight Compute profiling
├── agent/
│   ├── graph.py             # LangGraph optimization loop
│   └── store.py             # SQLite experiment tracking
├── scripts/
│   └── plot_results.py      # GFLOPS visualization
├── results/
│   └── gflops.png           # Benchmark comparison
├── Makefile
└── requirements.txt
```

## Running the Project

Experiments were run using Google Colab with an NVIDIA Tesla T4 GPU.

### 1. Clone and install

```bash
git clone https://github.com/charmikadi/cuda-kernel-tuning-agent.git
cd cuda-kernel-tuning-agent
pip install -r requirements.txt
```

### 2. Verify GPU and CUDA

```bash
nvidia-smi
nvcc --version
```

### 3. Compile and benchmark

```bash
make all
python -m bench.run_benchmark
```

### 4. Run the optimization agent

```bash
export OPENAI_API_KEY=your_api_key  # Optional
python -m agent.graph --iters 12
```

### 5. Generate results and profile

```bash
python -m scripts.plot_results
python -m bench.profile  # Requires Nsight Compute
```

**GPU architecture:** The default is `sm_75` for the NVIDIA Tesla T4. For an A100, use `sm_80`; for an RTX 30-series GPU, use `sm_86`.

## Tech Stack

**CUDA · C++ · Python · cuBLAS · LangGraph · OpenAI API · SQLite · Nsight Compute · Matplotlib**

## Future Improvements

- Explore warp-level primitives and more advanced memory optimizations.
- Incorporate Nsight Compute metrics directly into the agent's optimization decisions.
- Expand testing across different matrix sizes and NVIDIA GPU architectures.
