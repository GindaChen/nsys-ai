// Minimal CUDA axpy workload for local nsys-ai profile captures.
//
// Build (needs nvcc; a GPU is needed to run it, not to compile it):
//   nvcc -O2 -o /tmp/axpy scripts/axpy.cu
//
// Usage:
//   /tmp/axpy [launches]
//
// Launch count is a CLI argument (default 5) so one binary produces both a
// before and an after capture (e.g. 5 vs 40). See scripts/README.md.

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void axpy(float a, float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

static void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
}

static const long kMaxLaunches = 1000000;

int main(int argc, char** argv) {
    long launches = 5;
    if (argc >= 2) {
        char* end = nullptr;
        errno = 0;
        launches = std::strtol(argv[1], &end, 10);
        // errno/ERANGE matters: without it strtol saturates at LONG_MAX and every
        // other condition here passes, so an out-of-range argument becomes an
        // effectively unbounded launch loop under a capture with no timeout.
        if (end == argv[1] || *end != '\0' || errno == ERANGE
            || launches < 1 || launches > kMaxLaunches) {
            std::fprintf(stderr, "launches must be an integer in 1..%ld\n", kMaxLaunches);
            return 2;
        }
    }

    const int n = 1 << 20;
    const float a = 2.0f;
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    float* x = nullptr;
    float* y = nullptr;
    check(cudaMalloc(&x, n * sizeof(float)), "cudaMalloc x");
    check(cudaMalloc(&y, n * sizeof(float)), "cudaMalloc y");
    check(cudaMemset(x, 0, n * sizeof(float)), "cudaMemset x");
    // The kernel reads y before writing it; leaving it uninitialised would be an
    // uninitialised device read in a file that is otherwise fully error-checked.
    check(cudaMemset(y, 0, n * sizeof(float)), "cudaMemset y");

    for (long i = 0; i < launches; ++i) {
        axpy<<<blocks, threads>>>(a, x, y, n);
        check(cudaGetLastError(), "axpy launch");
    }
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    check(cudaFree(x), "cudaFree x");
    check(cudaFree(y), "cudaFree y");

    std::printf("axpy done: launches=%ld n=%d\n", launches, n);
    return 0;
}
