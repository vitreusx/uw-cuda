#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <cstring>
#include <vector>
#include <memory>
#include <stack>
#include <fstream>
#include <sstream>
using namespace std;

// Default numeric type.
typedef float real;

bool debugMode = false;

namespace cuda {
    // CUDA error-checking helper functions.
    void check(cudaError_t ret) {
        if (ret != cudaSuccess) {
            throw runtime_error(cudaGetErrorString(ret));
        }
    }

    void check() {
        check(cudaGetLastError());
    }

    // More malloc-like cudaMalloc
    template<typename T>
    T *malloc(size_t nelems) {
        T *res;
        cuda::check(cudaMalloc((void **)&res, sizeof(T) * nelems));
        return res;
    }

    // More free-like cudaFree
    void free(void *ptr) {
        cuda::check(cudaFree(ptr));
    }
}

// Time measurement facility.
class Timer {
private:
    struct Entry {
        time_t prior;
        string message;
    };

    stack<Entry> entries;

public:
    void enter(string msg) {
        Entry en;
        en.message = msg;

        entries.push(en);
        if (debugMode)
            cerr << "[" << msg << "] Starting timer." << '\n';

        entries.top().prior = clock();
    }

    real leave() {
        int after = clock();
        Entry en = entries.top();
        entries.pop();

        real dur = (real)(after - en.prior) / CLOCKS_PER_SEC;
        if (debugMode)
            cerr << "[" << en.message << "] Timing done: took " << dur << "s." << '\n';
        return dur;
    }
};

// Host-side numerical functions.
namespace host {
    // Compute <x, y> for x, y: R^dim
    real scalar(real *x, real *y, int dim) {
        real sum = 0;
        for (int i = 0; i < dim; ++i) sum += x[i] * y[i];
        return sum;
    }

    // Given {x_i} \subset R^dim of size nvec as src,
    // output {norm(x_i)} to dest
    void normalize(vector<real>& cnv, int dim, int nvec) {
        for (int i = 0; i < nvec; ++i) {
            real *x = cnv.data() + dim * i;
            real norm = sqrt(host::scalar(x, x, dim));
            for (int j = 0; j < dim; ++j) {
                *(x + j) /= norm;
            }
        }
    }

    // Comput similarity matrix (host-side).
    void similarity(real *cnv, real *sim, int dim, int nvec) {
        for (int i = 0; i < nvec; ++i) {
            real *x = cnv + dim * i;
            for (int j = i + 1; j < nvec; ++j) {
                real *y = cnv + dim * j;
                sim[i*nvec+j] = host::scalar(x, y, dim);
            }
        }
    }
}

// Kernels &c.
namespace device {
    template<size_t Nearest>
    __device__ void warpReduce(volatile real *aux, size_t tid) {
        if (Nearest >= 64) aux[tid] += aux[tid + 32];
        if (Nearest >= 32) aux[tid] += aux[tid + 16];
        if (Nearest >= 16) aux[tid] += aux[tid + 8];
        if (Nearest >=  8) aux[tid] += aux[tid + 4];
        if (Nearest >=  4) aux[tid] += aux[tid + 2];
        if (Nearest >=  2) aux[tid] += aux[tid + 1];
    }

    __device__ void deriveId(int m, int x, int y, volatile int *realx, volatile int *realy) {
        int ceil = (m - 1) / 2 + 1;

        if (m & 1) {
            if (x > y) {
                *realx = ceil + y;
                *realy = ceil + x;
            }
            else {
                *realx = x;
                *realy = y + 1;
            }
        }
        else {
            if (x >= y) {
                *realx = ceil + y;
                *realy = ceil + x + 1;
            }
            else {
                *realx = x;
                *realy = y;
            }
        }
    }

    // Compute <x, y> for x, y: R^dim and output into *res
    template<size_t Nearest>
    __global__ void scalar(real *mat, int dim, int nvec, real *res) {
        size_t tid = threadIdx.x;

        int realx, realy;
        deriveId(nvec - 1, blockIdx.x, blockIdx.y, &realx, &realy);
        real *x = &mat[dim * realx], *y = &mat[dim * realy];

        size_t lo = (tid * dim) / blockDim.x,
            hi = ((tid + 1) * dim) / blockDim.x;
        real total = 0;
        for (size_t i = lo; i < hi; ++i) {
            total += x[i] * y[i];
        }

        __shared__ real aux[Nearest];
        aux[tid] = total;
        if (tid + blockDim.x < Nearest) aux[tid + blockDim.x] = 0;
        __syncthreads();

        for (unsigned int s = Nearest / 2; s > 1024; s >>= 1) {
            if (tid < s) aux[tid] += aux[tid + s];
            __syncthreads();
        }

        if (Nearest >= 1024) {
            if (tid < 512) aux[tid] += aux[tid + 512];
            __syncthreads();
        }
        if (Nearest >= 512) {
            if (tid < 256) aux[tid] += aux[tid + 256];
            __syncthreads();
        }
        if (Nearest >= 256) {
            if (tid < 128) aux[tid] += aux[tid + 128];
            __syncthreads();
        }
        if (Nearest >= 128) {
            if (tid < 64) aux[tid] += aux[tid + 64];
            __syncthreads();
        }

        if (tid < 32) warpReduce<Nearest>(aux, tid);
        if (tid == 0) res[nvec * realx + realy] = aux[0];
    }

    // Compute similarity matrix for given CNV into sim,
    // using stream pool given with iterator.
    template<size_t Block, size_t Nearest>
    void similarity(real *cnv, real *sim, int dim, int nvec) {
        dim3 grid(nvec / 2, 2 * ((nvec - 1) / 2) + 1);
        device::scalar<Nearest>
            <<<grid, Block>>>
            (cnv, dim, nvec, sim);
    }
}

vector<real> readCSV(char *filename, int& dim, int& nvec) {
    size_t maxline = 2000000;
    unique_ptr<char, void(*)(void *)> buf {
        (char *)malloc(maxline),
        free
    };

    // Reserve CNV.
    vector<real> cnv;
    cnv.reserve(dim * nvec);

    ifstream file(filename);
    size_t linum = 0;
    while (file.getline(buf.get(), maxline)) {
        // Skip 1st line (i.e. the header)
        if (++linum > 1) {
            size_t colnum = 0;
            char *cur = strtok(buf.get(), ",");
            while (cur) {
                // Skip 1st column; push values in flat form.
                if (++colnum > 1) cnv.push_back(atof(cur));
                cur = strtok(nullptr, ",");
            }
        }
    }

    // Return dimensions.
    nvec = linum - 1;
    dim = cnv.size() / nvec;

    return cnv;
}

// Compute avg, min and max of similarity matrix sim: R^{nvec,nvec}
void reportStats(real *sim, int nvec) {
    real total = 0, max = 0, min = 1;

    for (int i = 0; i < nvec; ++i) {
        for (int j = i + 1; j < nvec; ++j) {
            real cur = sim[i*nvec+j] * sim[i*nvec+j];
            if (cur < min) min = cur;
            if (cur > max) max = cur;
            total += cur;
        }
    }

    total /= (nvec * (nvec - 1)) / 2;
    cerr << "Avg similarity: " << 100 * total << "%\n";
    cerr << "Min similarity: " << 100 * min << "%\n";
    cerr << "Max similarity: " << 100 * max << "%\n";
}

// Compute Pearson correlation coefficient (i.e. cov(X, Y)/[std(X)std(Y])
// for samples X, Y: R^dim
real correlation(real *x, real *y, int nvec) {
    real xsum, ysum, xysum, xxsum, yysum;
    xsum = ysum = xysum = xxsum = yysum = 0;

    for (int i = 0; i < nvec; ++i) {
        for (int j = i + 1; j < nvec; ++j) {
            int id = i * nvec + j;
            real xi = x[id], yi = y[id];
            xsum += xi;
            ysum += yi;
            xysum += xi * yi;
            xxsum += xi * xi;
            yysum += yi * yi;
        }
    }

    int dim = (nvec * (nvec - 1)) / 2;
    real xmean = xsum / dim, ymean = ysum / dim;
    real covxy = xysum - dim * xmean * ymean;
    real varx = xxsum - dim * xmean * xmean;
    real vary = yysum - dim * ymean * ymean;
    return covxy / sqrt(varx * vary);
}

template<size_t Block, size_t Nearest>
void computeFor(real *cnv, real *sim, real *simLocal, real *ref,
                int dim, int nvec,
                Timer& t) {
    stringstream ss {};
    ss << "Similarity (Device -"
       << " BlockSize = " << Block
       << ")";

    t.enter(ss.str());
    device::similarity<Block, Nearest>(cnv, sim, dim, nvec);
    cuda::check(cudaMemcpy(simLocal, sim,
                           nvec * nvec * sizeof(real),
                           cudaMemcpyDeviceToHost));
    real dur = t.leave();

    cout << "BLOCKSIZE " << Block
         << " TIME " << dur << '\n';

    if (debugMode) {
        reportStats(simLocal, nvec);
        real corr = correlation(simLocal, ref, nvec);
        cerr << "Correlation: " << corr << '\n';
    }
}

int main(int argc, char **argv) {
    char *filename = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (!debugMode && (!strcmp(argv[i], "-g") || !strcmp(argv[i], "--debug"))) {
            debugMode = true;
        } else if (!filename) {
            filename = argv[i];
        }
    }

    if (!filename) {
        cout << "Wrong number of arguments\n";
        cout << "Usage: " << argv[0] << " [-g|--debug] filename\n";
        cout << "Exiting\n";
        return EXIT_FAILURE;
    }

    Timer t;

    t.enter("Reading & parsing CSV file");
    int nvec = 150, dim = 50000;
    vector<real> cnv = readCSV(filename, dim, nvec);
    t.leave();

    t.enter("Normalizing CNV");
    host::normalize(cnv, dim, nvec);
    t.leave();

    t.enter("Allocating similarity matrices");
    unique_ptr<real, void(*)(void *)> simHost {
        (real *)malloc(nvec * nvec * sizeof(real)),
        free
    };

    unique_ptr<real, void(*)(void *)> simDev {
        cuda::malloc<real>(nvec * nvec),
        cuda::free
    };

    unique_ptr<real, void(*)(void *)> simDevLocal {
        (real *)malloc(nvec * nvec * sizeof(real)),
        free
    };
    t.leave();

    t.enter("Allocating & copying CNV to the device");
    unique_ptr<real, void(*)(void *)> cnvDev {
        cuda::malloc<real>(nvec * dim),
        cuda::free
    };

    cuda::check(cudaMemcpy(cnvDev.get(), cnv.data(),
                           nvec * dim * sizeof(real),
                           cudaMemcpyHostToDevice));
    t.leave();

    if (debugMode) {
        t.enter("Similarity (Host)");
        host::similarity(cnv.data(), simHost.get(), dim, nvec);
        reportStats(simHost.get(), nvec);
        t.leave();
    }

    computeFor<  32,  32>(cnvDev.get(), simDev.get(), simDevLocal.get(),
                          simHost.get(), dim, nvec, t);
    computeFor<  64,  64>(cnvDev.get(), simDev.get(), simDevLocal.get(),
                          simHost.get(), dim, nvec, t);
    computeFor<  96, 128>(cnvDev.get(), simDev.get(), simDevLocal.get(),
                          simHost.get(), dim, nvec, t);
    computeFor< 128, 128>(cnvDev.get(), simDev.get(), simDevLocal.get(),
                          simHost.get(), dim, nvec, t);
    computeFor< 192, 256>(cnvDev.get(), simDev.get(), simDevLocal.get(),
                          simHost.get(), dim, nvec, t);
    computeFor< 256, 256>(cnvDev.get(), simDev.get(), simDevLocal.get(),
                          simHost.get(), dim, nvec, t);
    computeFor< 320, 512>(cnvDev.get(), simDev.get(), simDevLocal.get(),
                          simHost.get(), dim, nvec, t);
    computeFor< 384, 512>(cnvDev.get(), simDev.get(), simDevLocal.get(),
                          simHost.get(), dim, nvec, t);
    computeFor< 448, 512>(cnvDev.get(), simDev.get(), simDevLocal.get(),
                          simHost.get(), dim, nvec, t);
    computeFor< 512, 512>(cnvDev.get(), simDev.get(), simDevLocal.get(),
                          simHost.get(), dim, nvec, t);
    computeFor< 640,1024>(cnvDev.get(), simDev.get(), simDevLocal.get(),
                          simHost.get(), dim, nvec, t);
    computeFor< 768,1024>(cnvDev.get(), simDev.get(), simDevLocal.get(),
                          simHost.get(), dim, nvec, t);
    computeFor< 896,1024>(cnvDev.get(), simDev.get(), simDevLocal.get(),
                          simHost.get(), dim, nvec, t);
    computeFor<1024,1024>(cnvDev.get(), simDev.get(), simDevLocal.get(),
                          simHost.get(), dim, nvec, t);

    return EXIT_SUCCESS;
}
