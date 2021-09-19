#pragma once
#include <cuda_runtime.h>
#include "const.h"

namespace dev {
    template<int Block>
    __device__ void warpReduce(volatile real *aux, int tid) {
        if (Block >= 64) aux[tid] += aux[tid + 32];
        if (Block >= 32) aux[tid] += aux[tid + 16];
        if (Block >= 16) aux[tid] += aux[tid + 8];
        if (Block >=  8) aux[tid] += aux[tid + 4];
        if (Block >=  4) aux[tid] += aux[tid + 2];
        if (Block >=  2) aux[tid] += aux[tid + 1];
    }

    template<int Block>
    __global__ void scalar(real *x, real *y, int dim, real *res) {
        __shared__ real aux[Block];

        real total = 0;
        int tid = threadIdx.x;
        for (int i = tid; i < dim; i += Block) {
            total += x[i] * y[i];
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

        if (tid == 0)
            *res = aux[0];
    }

    template<int Block>
    __global__ void sum(real *x, int dim, real *res) {
        __shared__ real aux[Block];

        real total = 0;
        int tid = threadIdx.x;
        for (int i = tid; i < dim; i += Block) {
            total += x[i];
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

        if (tid == 0)
            *res = aux[0];
    }
}
