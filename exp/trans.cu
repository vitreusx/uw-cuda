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
#include "utils/scalar.cuh"
using namespace std;

void trans_host(real *A, dim3 sh, real *AT) {
    for (int i = 0; i < sh.x; ++i) {
        for (int j = 0; j < sh.y; ++j) {
            AT[j * sh.x + i] = A[i * sh.y + j];
        }
    }
}

__global__ void trans_naive(real *A, dim3 sh, real *AT) {
    for (int i = threadIdx.x; i < sh.y; i += blockDim.x) {
        AT[sh.x * i + blockIdx.x] = A[sh.y * blockIdx.x + i];
    }
}

__global__ void trans_shared(real *A, dim3 sh, dim3 ntiles, real *AT) {
    assert(blockDim.y == WARP);

    __shared__ real partial[WARP][WARP + 1];
    int u, v, lim_x, lim_y, x, y, i;
    u = WARP * (ntiles.x * blockIdx.x);

    for (x = 0; x < ntiles.x && u < sh.x; ++x, u += WARP) {
        lim_x = min(WARP, sh.x - u);
        v = WARP * (ntiles.y * blockIdx.y);

        for (y = 0; y < ntiles.y && v < sh.y; ++y, v += WARP) {
            lim_y = min(WARP, sh.y - v);

            if (threadIdx.y < lim_y) {
                for (i = threadIdx.x; i < lim_x; i += blockDim.x) {
                    partial[i][threadIdx.y] = A[(u + i) * sh.y + (v + threadIdx.y)];
                }
            }
            __syncthreads();

            if (threadIdx.y < lim_x) {
                for (i = threadIdx.x; i < lim_y; i += blockDim.x) {
                    AT[(v + i) * sh.x + (u + threadIdx.y)] = partial[threadIdx.y][i];
                }
            }
            __syncthreads();
        }
    }
}

__global__ void trans_comp(real *A, dim3 sh, dim3 ntiles, real *AT) {
    assert(blockDim.y == WARP);

    real Apart[WARP];
    int u, v, lim_x, lim_y, x, i;

    for (x = threadIdx.x; x < ntiles.x * ntiles.y; x += blockDim.x) {
        u = WARP * (ntiles.x * blockIdx.x + x / ntiles.y);
        v = WARP * (ntiles.y * blockIdx.y + x % ntiles.y);
        lim_x = min(WARP, sh.x - u);
        lim_y = min(WARP, sh.y - v);
        
        #pragma unroll
        for (i = 0; i <= WARP; ++i) {
            if (u + i < lim_x && v + threadIdx.y < lim_y)
                Apart[i] = A[(u + i) * sh.y + (v + threadIdx.y)];
        }
        __syncwarp();

        #pragma unroll
        for (i = 0; i <= WARP; ++i) {
            if (v + i < lim_y && u + threadIdx.y < lim_x)
                AT[(v + i) * sh.x + (u + threadIdx.y)] = Apart[i];
        }
    }
}

__global__ void trans_noswap(real *A, dim3 sh, dim3 ntiles, real *AT) {
    assert(blockDim.y == WARP);

    real Apart[WARP];
    int u, v, lim_x, lim_y, x, i, j;

    for (x = threadIdx.x; x < ntiles.x * ntiles.y; x += blockDim.x) {
        u = WARP * (ntiles.x * blockIdx.x + x / ntiles.y);
        v = WARP * (ntiles.y * blockIdx.y + x % ntiles.y);
        lim_x = min(WARP, sh.x - u);
        lim_y = min(WARP, sh.y - v);

        j = threadIdx.y;
        for (i = 0; i <= threadIdx.y; ++i, --j) {
            if (i < lim_x && j < lim_y)
                Apart[i] = A[(u + i) * sh.y + (v + j)];
        }

        j = WARP - 1;
        for (; i < WARP; ++i, --j) {
            if (i < lim_x && j < lim_y)
                Apart[i] = A[(u + i) * sh.y + (v + j)];
        }
        __syncthreads();

        j = threadIdx.y;
        for (i = 0; i <= threadIdx.y; ++i, --j) {
            if (i < lim_y && j < lim_x)
                AT[(v + i) * sh.x + (u + j)] = Apart[j];
        }

        j = WARP - 1;
        for (; i < WARP; ++i, --j) {
            if (i < lim_y && j < lim_x)
                AT[(v + i) * sh.y + (u + j)] = Apart[j];
        }
        __syncthreads();
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

int main(int argc, char **argv) {
    Timer tim;

    dim3 sh(20000, 20000);
    int len = sh.x * sh.y;
    DevBuffer<real> Adev(len), ATdev(len);

    srand(time(NULL));
    int off = rand() % (len / 2);
    int span = 10;

    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
    curandSetPseudoRandomGeneratorSeed(prng, clock());
    curandGenerateUniform(prng, Adev, len);

    for (int i = 0; i < 10; ++i) {
        tim.enter();
        trans_naive<<<sh.x, 192>>>(Adev, sh, ATdev);
        check();
        tim.leave();
    }
    cout << "Naive\n";
    cout << "Time: " << tim.stats() << '\n';
    excerpt_dev(ATdev, off, span);

    DevBuffer<real> ATref(len);
    check(cudaMemcpy(ATref, ATdev, len * sizeof(real), cudaMemcpyDeviceToDevice));

    dim3 block(6, WARP);
    dim3 ntiles(6, 8);
    dim3 grid(sh.x / (WARP * ntiles.x) + 1, sh.y / (WARP * ntiles.y) + 1);

    check(cudaMemset(ATdev, 0, len * sizeof(real)));
    for (int i = 0; i < 50; ++i) {
        tim.enter();
        trans_shared<<<grid, block>>>(Adev, sh, ntiles, ATdev);
        check();
        tim.leave();
    }
    cout << "Shared\n";
    cout << "Time: " << tim.stats() << '\n';
    cout << "Corr: " << dev::corr(ATref, ATdev, len) << '\n';
    excerpt_dev(ATdev, off, span);

    check(cudaMemset(ATdev, 0, len * sizeof(real)));
    for (int i = 0; i < 5; ++i) {
        tim.enter();
        trans_comp<<<grid, block>>>(Adev, sh, ntiles, ATdev);
        check();
        tim.leave();
    }
    cout << "Time Comp\n";
    cout << "Time: " << tim.stats() << '\n';
    cout << "Corr: " << dev::corr(ATref, ATdev, len) << '\n';
    excerpt_dev(ATdev, off, span);

    check(cudaMemset(ATdev, 0, len * sizeof(real)));
    for (int i = 0; i < 5; ++i) {
        tim.enter();
        trans_noswap<<<grid, block>>>(Adev, sh, ntiles, ATdev);
        check();
        tim.leave();
    }
    cout << "No Swap\n";
    cout << "Time: " << tim.stats() << '\n';
    cout << "Corr: " << dev::corr(ATref, ATdev, len) << '\n';
    excerpt_dev(ATdev, off, span);

    return 0;
}