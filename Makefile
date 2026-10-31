ARCH ?= sm_75

naive:
	nvcc -O3 -arch=$(ARCH) -DKERNEL=0 kernels/gemm.cu -lcublas -o build/gemm_naive
tiled:
	nvcc -O3 -arch=$(ARCH) -DKERNEL=1 -DTILE=32 kernels/gemm.cu -lcublas -o build/gemm_tiled
coarsened:
	nvcc -O3 -arch=$(ARCH) -DKERNEL=2 -DTILE=64 -DCOARSEN=8 kernels/gemm.cu -lcublas -o build/gemm_coarsened
all: naive tiled coarsened
agent:
	python -m agent.graph --iters 12
plot:
	python -m scripts.plot_results
clean:
	rm -f build/gemm_* results/experiments.db results/gflops.png
