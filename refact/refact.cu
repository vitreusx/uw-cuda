#include <__clang_cuda_device_functions.h>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
using namespace std;

typedef float real;

void verify(cudaError_t ret) {
    if (ret != cudaSuccess) {
        throw runtime_error(cudaGetErrorString(ret));
    }
}

void check() {
    verify(cudaGetLastError());
}

template <typename T> class DevBuffer {
private:
    T *ptr;

public:
    DevBuffer() {
        ptr = NULL;
    }

    DevBuffer(int nelems) {
        verify(cudaMalloc((void **)&ptr, sizeof(T) * nelems));
    }

    ~DevBuffer() {
        if (ptr)
            verify(cudaFree(ptr));
    }

    operator T *() {
        return ptr;
    }
};

template <typename T> class HostBuffer {
private:
    T *ptr;

public:
    HostBuffer() {
        ptr = NULL;
    }

    HostBuffer(int nelems) {
        verify(cudaMallocHost((void **)&ptr, sizeof(T) * nelems));
    }

    ~HostBuffer() {
        if (ptr)
            verify(cudaFreeHost(ptr));
    }

    operator T *() {
        return ptr;
    }
};

class Timer {
public:
    class Frame {
    private:
        time_t prior;
        string message;
        vector<real> times;
        bool single;

    public:
        Frame() {}

        Frame(string msg, bool _single) {
            message = msg;
            single = _single;
            if (single)
                enter();
        }

        void enter() {
            prior = clock();
        }

        void leave() {
            time_t after = clock();
            real dur = (real)(after - prior) / CLOCKS_PER_SEC;
            times.push_back(dur);
        }

        void resolve(real &total, real &mean, real &std) {
            if (single)
                leave();

            cerr << "[" << message << "] Measurement complete.\n";
            total = 0;
            for (int i = 0; i < times.size(); ++i) {
                total += times[i];
            }

            mean = total / times.size();
            std = 0;
            for (int i = 0; i < times.size(); ++i) {
                std += (times[i] - mean) * (times[i] - mean);
            }
            if (times.size() > 1)
                std /= times.size() - 1;
            std = sqrt(std);

            if (times.size() >= 1) {
                cerr << "\t    N: " << times.size() << "\n"
                    << "\tTotal: " << total << "s\n";
            }
            if (times.size() > 1) {
                cerr << "\t Mean: " << mean << "s\n"
                    << "\t  Std: " << std << "s\n";
            }
        }

        void resolve() {
            real total, mean, std;
            resolve(total, mean, std);
        }
    };

    Frame measure(string msg, bool single = true) {
        cerr << "[" << msg << "] Measuring.\n";
        return Frame(msg, single);
    }
};

class Stream {
private:
    cudaStream_t stream;

public:
    Stream() {
        verify(cudaStreamCreate(&stream));
    }

    ~Stream() {
        verify(cudaStreamDestroy(stream));
    }

    inline operator cudaStream_t() {
        return stream;
    }
}

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

    template<int Block>
    __global__ void normalize(real *mat, int off, int nvec, int dim) {
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

        real norm = aux[0];
        for (int i = lo; i < hi; ++i) {
            vect[i] /= norm;
        }
    }

    __global__ void scalar(real *mat, int dim, int nvec, real *res) {
        // We skip sectors below the diagonal.
        if (blockIdx.x * blockDim.x > (blockIdx.y + 1) * blockDim.y)
            return;

        int xid = blockIdx.x * blockDim.x + threadIdx.x;
        real *x = xid < nvec ? &mat[xid * dim] : NULL;

        int yid = blockIdx.y * blockDim.y + threadIdx.y;
        real *y = yid < nvec ? &mat[yid * dim] : NULL;

        __shared__ real left[32][33], right[32][33];
        real sum = 0;
        for (int i = 0; i < dim; i += blockDim.x) {
            if (x && i + threadIdx.y < dim)
                left[threadIdx.x][threadIdx.y] = x[i + threadIdx.y];
            if (y && i + threadIdx.x < dim)
                right[threadIdx.x][threadIdx.y] = y[i + threadIdx.x];
            __syncthreads();
            
            for (int j = 0; j < blockDim.x && i + j < dim; ++j) {
                sum += left[threadIdx.x][j] * right[j][threadIdx.y];
            }
            __syncthreads();
        }

        if (x && y)
            res[xid * nvec + yid] = sum;
    }
}

class Program {
private:
    string filename;
    bool debugMode;
    Timer t;

    int nvec, dim;
    HostBuffer<real> cnvHost, simHost;
    DevBuffer<real> cnvDev, simDev;
    
    void retrieveData() {
        ifstream file(filename.c_str());

        file.seekg(0, ios::end);
        int nbytes = file.tellg();
        file.seekg(0, ios::beg);

        HostBuffer<char> filebuf(nbytes);
        file.read(filebuf, nbytes);

        char *cur = filebuf;
        while (*cur != '\n') ++cur;
        cur = strtok(cur, ",\n");
        
        dim = nvec = 0;
        int i = 0;
        
        cnvHost = HostBuffer<real>(145 * 45000);
        while (cur) {
            if (*cur != '\"') cnvHost[i++] = atof(cur);
            else ++nvec;
            cur = strtok(NULL, ",\n");
        }
        dim = i / nvec;
    }

    void normalize() {
        int chunk = 8;
        int nbytes = chunk * dim * sizeof(real);
        vector<Stream> streams(nvec / chunk + 1);

        for (int i = 0, off = 0; off < nvec; ++i, off += chunk) {
            cudaMemcpyAsync(&cnvDev[off * dim], &cnvHost[off + dim], nbytes, cudaMemcpyHostToDevice, streams[i]);
            device::normalize<256><<<chunk, 256, 0, streams[i]>>>(cnvDev, off, nvec, dim);
        }
    }

    void similarity() {
        simDev = DevBuffer<real>(nvec * nvec);
        dim3 block(8, 8);
        dim3 grid(nvec / block.x + 1, nvec / block.y + 1);
        device::scalar<<<grid, block>>>(cnvDev, dim, nvec, simDev);

        int nbytes = nvec * nvec * sizeof(real);
        verify(cudaMemcpy(simHost, simDev, nbytes, cudaMemcpyDeviceToHost));
    }

public:
    Program(int argc, char **argv) {
        char *filenamePtr = NULL;
        debugMode = false;

        for (int i = 1; i < argc; ++i) {
            if (!debugMode && (!strcmp(argv[i], "-g") || !strcmp(argv[i], "--debug"))) 
                debugMode = true;
            else if (!filenamePtr)
                filenamePtr = argv[i];
        }

        if (!filenamePtr) throw invalid_argument("argv");
        else filename = filenamePtr;

        if (!debugMode)
            cerr.setstate(ios::failbit);
    }

    int run() {
        retrieveData();
        return EXIT_SUCCESS;
    }
};

int main(int argc, char **argv) {
    try {
        return Program(argc, argv).run();
    }
    catch (invalid_argument) {
        cout << "Wrong number of arguments\n";
        cout << "Usage: " << argv[0] << " [-g|--debug] filename\n";
        cout << "Exiting\n";
        return EXIT_FAILURE;
    } 
    catch (exception &e) {
        cerr.clear();
        cerr << "[ERROR] Message: " << e.what() << endl;
        return EXIT_FAILURE;
    }
}
