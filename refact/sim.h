#pragma once
#include "defs.h"
#include <cuda_runtime.h>

namespace host {
    void sim(real *mat, int nvec, int dim, real *sim);
}

namespace dev {
    template<int X, int Y>
    __global__ void sim_ker(real *mat, int nvec, int dim, real *sim) {
        if (blockIdx.x * X > (blockIdx.y + 1) * Y)
            return;
            
        __shared__ real left[X][WARP + 1], right[Y][WARP + 1];
        real sum = 0.0f;
        const int tidx = threadIdx.y * X + threadIdx.x;
        const int warpidx = tidx / WARP, localidx = tidx % WARP;
        const int effx = min(X * (blockIdx.x + 1), nvec) - X * blockIdx.x;
        const int effy = min(Y * (blockIdx.y + 1), nvec) - Y * blockIdx.y;
        int j;
        real *xoff = &mat[X * blockIdx.x * dim + localidx];
        real *yoff = &mat[Y * blockIdx.y * dim + localidx];

        for (int i = 0; i < dim; i += WARP, xoff += WARP, yoff += WARP) {
            if (warpidx < (X * Y) / WARP && i + localidx < dim) {
                for (j = warpidx; j < effx; j += (X * Y) / WARP) {
                    left[j][localidx] = xoff[dim * j];
                }
                for (j = warpidx; j < effy; j += (X * Y) / WARP) {
                    right[j][localidx] = yoff[dim * j];
                }
            }
            __syncthreads();

            if (threadIdx.x < effx && threadIdx.y < effy) {
                for (j = 0; j < WARP && i + j < dim; ++j) {
                    sum += left[threadIdx.x][j] * right[threadIdx.y][j];
                }
            }
            __syncthreads();
        }

        if (threadIdx.x < effx && threadIdx.y < effy) {
            const int xidx = blockDim.x * blockIdx.x + threadIdx.x;
            const int yidx = blockDim.y * blockIdx.y + threadIdx.y;
            sim[xidx * nvec + yidx] = sim[yidx * nvec + xidx] = sum;
        }
    }
}
