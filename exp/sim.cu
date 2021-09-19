#include <assert.h>
#include <cuda_runtime.h>
#include <iostream>
#include <curand.h>
#include "utils/const.h"
#include "utils/metrics.cuh"
#include "utils/hostbuffer.cuh"
#include "utils/devbuffer.cuh"
#include "utils/utils.cuh"
#include "utils/timer.h"
#include "utils/cnv.h"
using namespace std;

static real dot(real *x, real *y, int dim) {
    real s = 0;
    for (int i = 0; i < dim; ++i, ++x, ++y) {
        s += *x * *y;
    }
    return s;
}

void sim_host(real *mat, int nvec, int dim, real *sim) {
    real *x = mat;
    for (int i = 0; i < nvec; ++i, x += dim) {
        real *y = x;
        for (int j = i; j < nvec; ++j, y += dim) {
            sim[i*nvec+j] = sim[j*nvec+i] = dot(x, y, dim);
        }
    }
}

void norm_host(real *mat, int nvec, int dim) {
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

template<int X, int Y>
__global__ void sim_ker1(real *mat, int shx, int shy, real *sim) {
    assert(blockDim.x == X && blockDim.y == Y);

    if (blockIdx.x * X > (blockIdx.y + 1) * Y)
        return;
        
    __shared__ real left[X][WARP + 1], right[Y][WARP + 1];
    real sum = 0.0f;
    const int tidx = threadIdx.y * X + threadIdx.x;
    const int warpidx = tidx / WARP, localidx = tidx % WARP;
    const int effx = min(blockDim.x, shx - blockDim.x * blockIdx.x);
    const int effy = min(blockDim.y, shx - blockDim.y * blockIdx.y);
    int j;
    real *xoff = &mat[X * blockIdx.x * shy + localidx];
    real *yoff = &mat[Y * blockIdx.y * shy + localidx];

    for (int i = 0; i < shy; i += WARP, xoff += WARP, yoff += WARP) {
        if (warpidx < (X * Y) / WARP && i + localidx < shy) {
            for (j = warpidx; j < effx; j += (X * Y) / WARP) {
                left[j][localidx] = xoff[shy * j];
            }
            for (j = warpidx; j < effy; j += (X * Y) / WARP) {
                right[j][localidx] = yoff[shy * j];
            }
        }
        __syncthreads();

        if (threadIdx.x < effx && threadIdx.y < effy) {
            for (j = 0; j < WARP && i + j < shy; ++j) {
                sum += left[threadIdx.x][j] * right[threadIdx.y][j];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x < effx && threadIdx.y < effy) {
        const int xidx = blockDim.x * blockIdx.x + threadIdx.x;
        const int yidx = blockDim.y * blockIdx.y + threadIdx.y;
        sim[xidx * shx + yidx] = sim[yidx * shx + xidx] = sum;
    }
}

int main(int argc, char **argv) {
    Timer tim;

    CNV A(argv[1]);
    dim3 sh(A.nvec, A.dim);
    int len = sh.x * sh.y;

    int slen = sh.x * sh.x;
    HostBuffer<real> Shost(slen);
    norm_host(A.data, sh.x, sh.y);
    sim_host(A.data, sh.x, sh.y, Shost);
    cout << "Host <Met>: " << host::stats(Shost, sh.x) << '\n';

    DevBuffer<real> Adev(len);
    check(cudaMemcpy(Adev, A.data, len * sizeof(real), cudaMemcpyHostToDevice));

    DevBuffer<real> S1(slen);
    HostBuffer<real> S1copy(slen);
    dim3 block(16, 16);
    dim3 grid(sh.x / block.x + 1, sh.x / block.y + 1);
    for (int i = 0; i < 50; ++i) {
        tim.enter();
        sim_ker1<16, 16><<<grid, block>>>(Adev, sh.x, sh.y, S1);
        check(cudaMemcpy(S1copy, S1, slen * sizeof(real), cudaMemcpyDeviceToHost));
        tim.leave();
    }
    cout << "Ker #1 <Time>: " << tim.stats() << '\n';
    cout << "Ker #1 <Met>: " << host::stats(S1copy, sh.x) << '\n';
    cout << "Host <-> Ker #1: " << host::corr(S1copy, Shost, slen) << '\n';
    return 0;
}
