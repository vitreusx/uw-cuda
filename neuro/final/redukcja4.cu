#include "redukcja.h"

__device__ void warpReduce(volatile real *aux, size_t tid) {
    aux[tid] += aux[tid + 32];
    aux[tid] += aux[tid + 16];
    aux[tid] += aux[tid + 8];
    aux[tid] += aux[tid + 4];
    aux[tid] += aux[tid + 2];
    aux[tid] += aux[tid + 1];
}

// <<Unroll last warp>>
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

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) aux[tid] += aux[tid + s];
        __syncthreads();
    }

    if (tid < 32) warpReduce(aux, tid);
    if (tid == 0) *res = aux[0];
}
