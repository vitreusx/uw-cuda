#include "normalize.h"

template<size_t Block>
__device__ void warpReduce(volatile real *aux, size_t tid) {
    if (Block >= 64) aux[tid] += aux[tid + 32];
    if (Block >= 32) aux[tid] += aux[tid + 16];
    if (Block >= 16) aux[tid] += aux[tid + 8];
    if (Block >=  8) aux[tid] += aux[tid + 4];
    if (Block >=  4) aux[tid] += aux[tid + 2];
    if (Block >=  2) aux[tid] += aux[tid + 1];
}

template<int Block>
__global__ void dev::norm_ker(real *mat, int off, int nvec, int dim) {
    // By structure, each block is responsible for normalizing one vect.
    // In phase 1, we reduce the vector; in phase 2 we divide elements by the norm.

    if (off + blockIdx.x >= nvec) return;

    __shared__ real aux[Block];
    real *vect = &mat[(off + blockIdx.x) * dim];
    int tid = threadIdx.x,
        lo = (tid * dim) / blockDim.x,
        hi = ((tid + 1) * dim) / blockDim.x;

    real total = 0;
    for (int i = lo; i < hi; ++i) {
        total += vect[i] * vect[i];
    }

    aux[tid] = total;
    __syncthreads();

    for (int s = Block / 2; s > 1024; s >>= 1) {
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
    __syncthreads();
        
    real norm = sqrtf(aux[0]);
    for (int i = lo; i < hi; ++i) {
        vect[i] /= norm;
    }
}

void host::norm(real *mat, int nvec, int dim) {
    for (int i = 0; i < nvec; ++i) {
        real norm = 0;
        for (int j = 0; j < dim; ++j) {
            auto r = mat[dim * i + j];
            norm += r * r;
        }

        norm = sqrt(norm);
        for (int j = 0; j < dim; ++j) {
            mat[dim * i + j] /= norm;
        }
    }
}
