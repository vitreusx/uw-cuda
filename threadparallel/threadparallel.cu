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
            for (int j = 0; j < nvec; ++j) {
                real *y = cnv + dim * j;
                sim[i*nvec+j] = host::scalar(x, y, dim);
            }
        }
    }
}

// Kernels &c.
namespace device {
    // Compute <x, y> for x, y: R^dim and output into *res
    __global__ void scalar(real *mat, int dim, int nvec, real *res) {
        size_t xidx = blockIdx.x * blockDim.x + threadIdx.x;
        if (xidx >= nvec) return;

        size_t yidx = blockIdx.y * blockDim.y + threadIdx.y;
        if (yidx >= nvec) return;

        real *x = &mat[dim * xidx], *y = &mat[dim * yidx];
        real total = 0;
        for (int i = 0; i < dim; ++i) {
            total += x[i] * y[i];
        }

        res[xidx * nvec + yidx] = total;
    }

    // Compute similarity matrix for given CNV into sim,
    // using stream pool given with iterator.
    void similarity(real *cnv, real *sim, dim3 block, int dim, int nvec) {
        dim3 grid(nvec / block.x + 1, nvec / block.y + 1);
        device::scalar
            <<<grid, block>>>
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

    vector<int> sides = { 4, 8, 12, 16, 20, 24, 28, 32 };
    for (int x: sides) {
        for (int y: sides) {
            if (x != 4 || y != 4) {
                stringstream ss {};
                ss << "Similarity (Device -"
                   << " X = " << x
                   << "; Y = " << y
                   << ")";

                dim3 block(x, y);

                t.enter(ss.str());
                device::similarity(cnvDev.get(), simDev.get(), block, dim, nvec);
                cuda::check(cudaMemcpy(simDevLocal.get(), simDev.get(),
                                       nvec * nvec * sizeof(real),
                                       cudaMemcpyDeviceToHost));
                real dur = t.leave();

                cout << "X " << x << " Y " << y
                     << " TIME " << dur << '\n';

                if (debugMode) {
                    reportStats(simDevLocal.get(), nvec);
                    real corr = correlation(simDevLocal.get(), simHost.get(), nvec * nvec);
                    cerr << "Correlation: " << corr << '\n';
                }
            }
        }
    }

    return EXIT_SUCCESS;
}
