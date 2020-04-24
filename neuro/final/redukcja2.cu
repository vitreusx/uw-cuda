#include "redukcja.h"

// <<Interleaved w/ bank conflicts>>
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

    for (unsigned int s = 1; s < blockDim.x; s <<= 1) {
        int id = 2 * s * tid;
        if (id < blockDim.x) aux[id] += aux[id + s];
        __syncthreads();
    }

    if (tid == 0) *res = aux[0];
}
