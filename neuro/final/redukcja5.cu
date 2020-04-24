#include "redukcja.h"

template<size_t Block>
__device__ void warpReduce(volatile real *aux, size_t tid) {
    if (Block >= 64) aux[tid] += aux[tid + 32];
    if (Block >= 32) aux[tid] += aux[tid + 16];
    if (Block >= 16) aux[tid] += aux[tid + 8];
    if (Block >=  8) aux[tid] += aux[tid + 4];
    if (Block >=  4) aux[tid] += aux[tid + 2];
    if (Block >=  2) aux[tid] += aux[tid + 1];
}

// <<Completely unrolled>>
template<size_t Block>
__global__ void device::scalar(real *x, real *y, int dim, real *res) {
    size_t tid = threadIdx.x;

    // Sum a sector for the thread.
    size_t lo = (tid * dim) / blockDim.x,
           hi = ((tid + 1) * dim) / blockDim.x;
    real total = 0;
    for (size_t i = lo; i < hi; ++i) {
        total += x[i] * y[i];
    }

    __shared__ real aux[Block];
    aux[tid] = total;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 1024; s >>= 1) {
        if (tid < s) aux[tid] += aux[tid + s];
        __syncthreads();
    }

    if (Block >= 1024) {
        if (tid < 512) aux[tid] += aux[tid + 512];
        __syncthreads();
    }
    if (Block >= 512) {
        if (tid < 256) aux[tid] += aux[tid + 256];
        __syncthreads();
    }
    if (Block >= 256) {
        if (tid < 128) aux[tid] += aux[tid + 128];
        __syncthreads();
    }
    if (Block >= 128) {
        if (tid < 64) aux[tid] += aux[tid + 64];
        __syncthreads();
    }

    if (tid < 32) warpReduce<Block>(aux, tid);
    if (tid == 0) *res = aux[0];
}
