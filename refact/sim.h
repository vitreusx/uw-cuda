#pragma once
#include "defs.h"
#include <cuda_runtime.h>

namespace host {
    void sim(real *mat, int nvec, int dim, real *sim);
}

namespace dev {
    template<int X, int Y>
    __global__ void sim_ker(real *mat, dim3 ext, dim3 off, int nvec, int dim, real *sim) {
        if (blockIdx.x * X > (blockIdx.y + 1) * Y)
            return;
            
        __shared__ real left[X][WARP + 1], right[Y][WARP + 1];
        real sum = 0.0f;
        const int tidx = threadIdx.y * X + threadIdx.x;
        const int warpidx = tidx / WARP, localidx = tidx % WARP;
        const int effx = min(X * (blockIdx.x + 1), ext.x) - X * blockIdx.x;
        const int effy = min(Y * (blockIdx.y + 1), ext.y) - Y * blockIdx.y;
        int j;
        real *xoff = &mat[(off.x + X * blockIdx.x) * dim + localidx];
        real *yoff = &mat[(off.y + Y * blockIdx.y) * dim + localidx];

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
            const int xidx = off.x + blockDim.x * blockIdx.x + threadIdx.x;
            const int yidx = off.y + blockDim.y * blockIdx.y + threadIdx.y;
            sim[xidx * nvec + yidx] = sim[yidx * nvec + xidx] = sum;
        }
    }

    template<int Y>
    __global__ void sim_ker1(real *mat, int nvec, int dim, real *sim) {
        if (blockIdx.x * blockDim.x > (blockIdx.y + 1) * blockDim.y)
            return;

        __shared__ real left[WARP][WARP + 1], right[Y][WARP + 1];
        real sum = 0;
        int j;

        int effx = min(blockDim.x, nvec - blockDim.x * blockIdx.x);
        int effy = min(blockDim.y, nvec - blockDim.y * blockIdx.y);

        real *xvec = &mat[WARP * blockIdx.x * dim + threadIdx.x];
        real *yvec = &mat[Y * blockIdx.y * dim + threadIdx.x];
        for (int i = 0; i < dim; i += WARP, xvec += WARP, yvec += WARP) {
            if (i + threadIdx.x < dim) {
                for (j = threadIdx.y; j < effx; j += Y) {
                    left[j][threadIdx.x] = xvec[dim * j];
                }
                for (j = threadIdx.y; j < effy; j += Y) {
                    right[j][threadIdx.x] = yvec[dim * j];
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
            int xidx = blockDim.x * blockIdx.x + threadIdx.x;
            int yidx = blockDim.y * blockIdx.y + threadIdx.y;
            sim[xidx * nvec + yidx] = sim[yidx * nvec + xidx] = sum;
        }
    }
}
