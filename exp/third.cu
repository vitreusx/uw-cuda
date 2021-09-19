#include <assert.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "utils/const.h"
#include "utils/metrics.cuh"
#include "utils/hostbuffer.cuh"
#include "utils/devbuffer.cuh"
#include "utils/utils.cuh"
#include "utils/timer.h"
#include "utils/cnv.h"
#include "utils/norm.cuh"
using namespace std;

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

__global__ void trans_shared(real *A, dim3 sh, dim3 ntiles, real *AT) {
    assert(blockDim.x == WARP);

    __shared__ real partial[WARP][WARP + 1];
    int u, v, lim_x, lim_y, x, y, i;
    u = WARP * (ntiles.x * blockIdx.x);

    for (x = 0; x < ntiles.x && u < sh.x; ++x, u += WARP) {
        lim_x = min(WARP, sh.x - u);
        v = WARP * (ntiles.y * blockIdx.y);

        for (y = 0; y < ntiles.y && v < sh.y; ++y, v += WARP) {
            lim_y = min(WARP, sh.y - v);

            if (threadIdx.x < lim_y) {
                for (i = threadIdx.y; i < lim_x; i += blockDim.y) {
                    partial[i][threadIdx.x] = A[(u + i) * sh.y + (v + threadIdx.x)];
                }
            }
            __syncthreads();

            if (threadIdx.x < lim_x) {
                for (i = threadIdx.y; i < lim_y; i += blockDim.y) {
                    AT[(v + i) * sh.x + (u + threadIdx.x)] = partial[threadIdx.x][i];
                }
            }
            __syncthreads();
        }
    }
}

#define NVEC 145
__global__ void full_package(real *A, int dim, real *S) {
    assert(blockDim.x == WARP);

    int u = WARP * blockIdx.x, v = WARP * blockIdx.y;
    if (u > v)
        return;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int a, b, i, j;
    real sum;

    __shared__ real L[NVEC][WARP+1], R[NVEC][WARP+1];

    __syncthreads();
    if (u + threadIdx.x < dim) {
        for (i = threadIdx.y; i < NVEC; i += blockDim.y) {
            L[i][threadIdx.x] = A[i * dim + (u + threadIdx.x)];
        }
    }
    if (v + threadIdx.x < dim) {
        for (i = threadIdx.y; i < NVEC; i += blockDim.y) {
            R[i][threadIdx.x] = A[i * dim + (v + threadIdx.x)];
        }
    }

    __syncthreads();
    for (i = tid; i < WARP * WARP; i += blockDim.x * blockDim.y) {
        b = i % WARP;
        a = (i / WARP + b) % WARP;

        if (u + a < dim && v + b < dim) {
            sum = 0;
            for (j = 0; j < NVEC; ++j) {
                sum += L[j][a] * R[j][b];
            }
            S[(u + a) * dim + (v + b)] = sum;
            S[(v + b) * dim + (u + a)] = sum;
        }
    }
}

void excerpt(real *buf, int off, int span) {
    cout << "Excerpt:";
    for (int i = 0; i < span; ++i) {
        cout << " " << buf[i];
    }
    cout << '\n';
}

void excerpt_dev(real *buf, int off, int span) {
    HostBuffer<real> local(span);
    check(cudaMemcpy(local, buf + off, span * sizeof(real), cudaMemcpyDeviceToHost));
    excerpt(local, 0, span);
}

template<int N>
void testPrior(dim3 sh, real *AT, real *S, Timer& tim) {
    tim.enter();

    auto block = dim3(N, N);
    auto grid = dim3(sh.y / block.x + 1, sh.y / block.y + 1);
    sim_ker1<N, N><<<grid, block>>>(AT, sh.y, sh.x, S);
    check();

    tim.leave();
    cout << "[Prior - " << N << "]: " << tim.stats() << '\n';
}

int main(int argc, char **argv) {
    Timer tim;

    CNV A(argv[1]);
    dim3 sh(A.nvec, A.dim);
    dim3 block, grid;
    int len = sh.x * sh.y;

    DevBuffer<real> Adev(len);
    check(cudaMemcpy(Adev, A.data, len * sizeof(real), cudaMemcpyHostToDevice));

    DevBuffer<real> AT(len);
    DevBuffer<real> S(sh.y * sh.y);

    block = dim3(WARP, 16);
    dim3 ntiles(8, 8);
    grid = dim3(sh.x / (WARP * ntiles.x) + 1, sh.y / (WARP * ntiles.y) + 1);
    trans_shared<<<grid, block>>>(Adev, sh, ntiles, AT);
    check();

    block = dim3(256);
    grid = dim3(sh.y);
    dev::norm_ker<256><<<grid, block>>>(AT, sh.x);
    check();

    block = dim3(WARP, 16);
    grid = dim3(sh.y / (WARP * ntiles.x) + 1, sh.x / (WARP * ntiles.y) + 1);
    trans_shared<<<grid, block>>>(AT, dim3(sh.y, sh.x), ntiles, Adev);
    check();

    testPrior<8>(sh, AT, S, tim);
    testPrior<12>(sh, AT, S, tim);
    testPrior<16>(sh, AT, S, tim);
    testPrior<20>(sh, AT, S, tim);
    testPrior<24>(sh, AT, S, tim);
    testPrior<28>(sh, AT, S, tim);
    testPrior<32>(sh, AT, S, tim);

    int excerpt = 10000000;
    DevBuffer<real> Exc1(excerpt);
    check(cudaMemcpy(Exc1, S + 1000000, excerpt * sizeof(real), cudaMemcpyDeviceToDevice));
    excerpt_dev(Exc1, 0, 10);
    check(cudaMemset(S, 0, len * sizeof(real)));

    vector<int> sizes = { 8, 12, 16, 20, 24, 28, 32 };
    for (auto& s: sizes) {
        tim.enter();

        block = dim3(WARP, s);
        grid = dim3(sh.y / WARP + 1, sh.y / WARP + 1);
        full_package<<<grid, block>>>(Adev, sh.y, S);
        check();
    
        tim.leave();
        cout << "[New - " << s << "]: " << tim.stats() << '\n';
    }    

    DevBuffer<real> Exc2(excerpt);
    check(cudaMemcpy(Exc2, S + 1000000, excerpt * sizeof(real), cudaMemcpyDeviceToDevice));
    excerpt_dev(Exc2, 0, 10);

    cout << "Corr: " << dev::corr(Exc1, Exc2, excerpt) << '\n';
    return 0;
}
