// Single-precision GEMM (C = A * B, row-major) with three kernel variants.
// The variant and its parameters are chosen at compile time so the agent can
// treat every configuration as a separate build:
//   -DKERNEL=0  naive (one thread per output)
//   -DKERNEL=1  shared-memory tiling            (uses TILE)
//   -DKERNEL=2  tiling + thread coarsening      (uses TILE, COARSEN)
// Prints one JSON line with runtime, GFLOPS, cuBLAS baseline, and max error.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifndef KERNEL
#define KERNEL 1
#endif
#ifndef TILE
#define TILE 32
#endif
#ifndef COARSEN
#define COARSEN 4
#endif

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while (0)

__global__ void gemm_naive(const float* A, const float* B, float* C, int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N) {
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
  }
}

template <int T>
__global__ void gemm_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
  __shared__ float As[T][T];
  __shared__ float Bs[T][T];
  int tx = threadIdx.x, ty = threadIdx.y;
  int row = blockIdx.y * T + ty;
  int col = blockIdx.x * T + tx;
  float acc = 0.f;
  for (int t = 0; t < (K + T - 1) / T; ++t) {
    int ak = t * T + tx, bk = t * T + ty;
    As[ty][tx] = (row < M && ak < K) ? A[row * K + ak] : 0.f;
    Bs[ty][tx] = (bk < K && col < N) ? B[bk * N + col] : 0.f;
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < T; ++k) acc += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = acc;
}

// Each thread computes CF consecutive rows of one output column.
template <int T, int CF>
__global__ void gemm_coarsened(const float* A, const float* B, float* C, int M, int N, int K) {
  __shared__ float As[T][T];
  __shared__ float Bs[T][T];
  constexpr int ROWS_PER_PASS = T / CF;      // blockDim.y
  int tx = threadIdx.x, ty = threadIdx.y;
  int rowBase = blockIdx.y * T;
  int col = blockIdx.x * T + tx;
  float acc[CF] = {0.f};
  for (int t = 0; t < (K + T - 1) / T; ++t) {
    #pragma unroll
    for (int i = 0; i < CF; ++i) {
      int r = ty + i * ROWS_PER_PASS;
      int ar = rowBase + r, ak = t * T + tx, bk = t * T + r;
      As[r][tx] = (ar < M && ak < K) ? A[ar * K + ak] : 0.f;
      Bs[r][tx] = (bk < K && col < N) ? B[bk * N + col] : 0.f;
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < T; ++k) {
      float b = Bs[k][tx];
      #pragma unroll
      for (int i = 0; i < CF; ++i) acc[i] += As[ty * CF + i][k] * b;
    }
    __syncthreads();
  }
  #pragma unroll
  for (int i = 0; i < CF; ++i) {
    int r = rowBase + ty * CF + i;
    if (r < M && col < N) C[r * N + col] = acc[i];
  }
}

static void launch(const float* A, const float* B, float* C, int M, int N, int K) {
#if KERNEL == 0
  dim3 block(16, 16), grid((N + 15) / 16, (M + 15) / 16);
  gemm_naive<<<grid, block>>>(A, B, C, M, N, K);
#elif KERNEL == 1
  dim3 block(TILE, TILE), grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  gemm_tiled<TILE><<<grid, block>>>(A, B, C, M, N, K);
#else
  static_assert(TILE % COARSEN == 0, "TILE must be divisible by COARSEN");
  dim3 block(TILE, TILE / COARSEN), grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  gemm_coarsened<TILE, COARSEN><<<grid, block>>>(A, B, C, M, N, K);
#endif
}

int main(int argc, char** argv) {
  int M = argc > 1 ? atoi(argv[1]) : 2048;
  int N = argc > 2 ? atoi(argv[2]) : 2048;
  int K = argc > 3 ? atoi(argv[3]) : 2048;
  int iters = argc > 4 ? atoi(argv[4]) : 20;

  std::vector<float> hA((size_t)M * K), hB((size_t)K * N), hC((size_t)M * N), hRef((size_t)M * N);
  srand(42);
  for (auto& x : hA) x = (rand() / (float)RAND_MAX) - 0.5f;
  for (auto& x : hB) x = (rand() / (float)RAND_MAX) - 0.5f;

  float *dA, *dB, *dC, *dRef;
  CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, hC.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dRef, hRef.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // Custom kernel: warm up, then time.
  launch(dA, dB, dC, M, N, K);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) launch(dA, dB, dC, M, N, K);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  ms /= iters;

  // cuBLAS baseline. cuBLAS is column-major, so compute C^T = B^T * A^T.
  cublasHandle_t handle;
  cublasCreate(&handle);
  const float alpha = 1.f, beta = 0.f;
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dRef, N);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dRef, N);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms_ref = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms_ref, start, stop));
  ms_ref /= iters;

  CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hRef.data(), dRef, hRef.size() * sizeof(float), cudaMemcpyDeviceToHost));
  double max_err = 0.0;
  for (size_t i = 0; i < hC.size(); ++i) max_err = fmax(max_err, fabs((double)hC[i] - hRef[i]));

  double flops = 2.0 * M * N * K;
  printf("{\"kernel\": %d, \"tile\": %d, \"coarsen\": %d, \"M\": %d, \"N\": %d, \"K\": %d, "
         "\"ms\": %.4f, \"gflops\": %.2f, \"cublas_ms\": %.4f, \"cublas_gflops\": %.2f, "
         "\"pct_of_cublas\": %.2f, \"max_abs_err\": %.6g, \"correct\": %s}\n",
         KERNEL, TILE, COARSEN, M, N, K, ms, flops / (ms * 1e6), ms_ref, flops / (ms_ref * 1e6),
         100.0 * ms_ref / ms, max_err, max_err < 1e-2 ? "true" : "false");

  cublasDestroy(handle);
  cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dRef);
  return 0;
}
