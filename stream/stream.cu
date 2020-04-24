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

    // Managed set of streams.
    class StreamSet {
    private:
        vector<cudaStream_t> streams;

    public:
        StreamSet(size_t max) {
            streams = vector<cudaStream_t>(max);
            for (size_t i = 0; i < max; ++i) {
                check(cudaStreamCreate(&streams[i]));
            }
        }

        ~StreamSet() {
            for (size_t i = 0; i < streams.size(); ++i) {
                check(cudaStreamDestroy(streams[i]));
            }
        }

        class Iterator {
        private:
            vector<cudaStream_t>& streams;
            size_t cur, max;

        public:
            Iterator(vector<cudaStream_t>& _streams, size_t _max)
                : streams (_streams) {
                cur = 0;
                max = _max;
            }

            cudaStream_t& operator*() {
                return streams[cur];
            }

            Iterator& operator++() {
                cur = (cur + 1) % max;
                return *this;
            }
        };

        Iterator pool(size_t nstreams) {
            if (nstreams <= streams.size()) return Iterator(streams, nstreams);
            else throw runtime_error("StreamSet::create");
        }
    };

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
    template<size_t Block>
    __device__ void warpReduce(volatile real *aux, size_t tid) {
        if (Block >= 64) aux[tid] += aux[tid + 32];
        if (Block >= 32) aux[tid] += aux[tid + 16];
        if (Block >= 16) aux[tid] += aux[tid + 8];
        if (Block >=  8) aux[tid] += aux[tid + 4];
        if (Block >=  4) aux[tid] += aux[tid + 2];
        if (Block >=  2) aux[tid] += aux[tid + 1];
    }

    // Compute <x, y> for x, y: R^dim and output into *res
    template<size_t Block>
    __global__ void scalar(real *x, real *y, int dim, real *res) {
        size_t tid = threadIdx.x;

        size_t lo = (tid * dim) / blockDim.x,
            hi = ((tid + 1) * dim) / blockDim.x;
        real total = 0;
        for (size_t i = lo; i < hi; ++i) {
            total += x[i] * y[i];
        }

        __shared__ real aux[Block];
        aux[tid] = total;
        __syncthreads();

        for (unsigned int s = Block / 2; s > 1024; s >>= 1) {
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
        if (tid == 0) *res = aux[0];
    }

    // Compute similarity matrix for given CNV into sim,
    // using stream pool given with iterator.
    template<size_t Block>
    void similarity(real *cnv, real *sim, int dim, int nvec,
                    cuda::StreamSet::Iterator iter) {
        for (int i = 0; i < nvec; ++i) {
            real *x = cnv + i * dim;
            for (int j = i + 1; j < nvec; ++j) {
                real *y = cnv + j * dim;
                device::scalar<Block>
                    <<<1, Block, 0, *iter>>>
                    (x, y, dim, &sim[i * nvec + j]);
                ++iter;
            }
        }
        cuda::check(cudaDeviceSynchronize());
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
real correlation(real *x, real *y, int dim) {
    real xsum, ysum, xysum, xxsum, yysum;
    xsum = ysum = xysum = xxsum = yysum = 0;

    for (int i = 0; i < dim; ++i) {
        real xi = x[i], yi = y[i];
        xsum += xi;
        ysum += yi;
        xysum += xi * yi;
        xxsum += xi * xi;
        yysum += yi * yi;
    }

    real xmean = xsum / dim, ymean = ysum / dim;
    real covxy = xysum - dim * xmean * ymean;
    real varx = xxsum - dim * xmean * xmean;
    real vary = yysum - dim * ymean * ymean;
    return covxy / sqrt(varx * vary);
}

template<size_t Block>
void computeFor(vector<size_t> const& streamSpec,
                cuda::StreamSet& streams,
                real *cnv, real *sim, real *simLocal, real *ref,
                int dim, int nvec,
                Timer& t) {
    for (size_t i = 0; i < streamSpec.size(); ++i) {
        size_t nstreams = streamSpec[i];

        stringstream ss {};
        ss << "Similarity (Device -"
           << " #Streams = " << nstreams
           << " BlockSize = " << Block
           << ")";

        cuda::StreamSet::Iterator iter = streams.pool(nstreams);

        t.enter(ss.str());
        device::similarity<Block>(cnv, sim, dim, nvec, iter);
        cuda::check(cudaMemcpy(simLocal, sim,
                               nvec * nvec * sizeof(real),
                               cudaMemcpyDeviceToHost));
        real dur = t.leave();

        cout << "BLOCKSIZE " << Block
             << " STREAMS " << nstreams
             << " TIME " << dur << '\n';

        if (debugMode) {
            reportStats(simLocal, nvec);
            real corr = correlation(simLocal, ref, nvec * nvec);
            cerr << "Correlation: " << corr << '\n';
        }
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
    vector<size_t> streamSpec =
        {1, 2, 4, 8, 16, 32, 64, 128};
    cuda::StreamSet streams(128);

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

    computeFor<  32>(streamSpec, streams,
                     cnvDev.get(), simDev.get(), simDevLocal.get(),
                     simHost.get(), dim, nvec, t);
    computeFor<  64>(streamSpec, streams,
                     cnvDev.get(), simDev.get(), simDevLocal.get(),
                     simHost.get(), dim, nvec, t);
    computeFor< 128>(streamSpec, streams,
                     cnvDev.get(), simDev.get(), simDevLocal.get(),
                     simHost.get(), dim, nvec, t);
    computeFor< 256>(streamSpec, streams,
                     cnvDev.get(), simDev.get(), simDevLocal.get(),
                     simHost.get(), dim, nvec, t);
    computeFor< 512>(streamSpec, streams,
                     cnvDev.get(), simDev.get(), simDevLocal.get(),
                     simHost.get(), dim, nvec, t);
    computeFor<1024>(streamSpec, streams,
                     cnvDev.get(), simDev.get(), simDevLocal.get(),
                     simHost.get(), dim, nvec, t);

    return EXIT_SUCCESS;
}
