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

template <typename T>
class DevBuffer {
private:
    T *ptr;

public:
    DevBuffer() {
        ptr = NULL;
    }

    DevBuffer(int nelems) {
        verify(cudaMalloc((void **)&ptr, sizeof(T) * nelems));
    }

    DevBuffer(DevBuffer const& other) = delete;
    DevBuffer& operator=(DevBuffer const& other) = delete;

    DevBuffer(DevBuffer&& other) {
        *this = move(other);
    }

    DevBuffer& operator=(DevBuffer&& other) {
        ptr = other.ptr;
        other.ptr = NULL;
        return *this;
    }

    ~DevBuffer() {
        if (ptr)
            verify(cudaFree(ptr));
    }

    operator T *() {
        return ptr;
    }
};

template <typename T>
class HostBuffer {
private:
    T *ptr;

public:
    HostBuffer() {
        ptr = NULL;
    }

    HostBuffer(int nelems) {
        verify(cudaMallocHost((void **)&ptr, sizeof(T) * nelems));
    }

    HostBuffer(HostBuffer const& other) = delete;
    HostBuffer& operator=(HostBuffer const& other) = delete;

    HostBuffer(HostBuffer&& other) {
        *this = move(other);
    }

    HostBuffer& operator=(HostBuffer&& other) {
        ptr = other.ptr;
        other.ptr = NULL;
        return *this;
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
};

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
        
        real norm = sqrtf(aux[0]);
        for (int i = lo; i < hi; ++i) {
            vect[i] /= norm;
        }
    }

    __global__ void similarity(real *mat, int dim, int nvec, real *res) {
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

        if (x && y) {
            res[xid * nvec + yid] = sum;
            res[yid * nvec + xid] = sum;
        }
    }
}

class Program {
private:
    string filename;
    bool debugMode;
    Timer t;
    Timer::Frame fr;

    int nvec, dim;
    HostBuffer<real> cnvOnHost, simOnHost, cnvFromDev, simFromDev;
    DevBuffer<real> cnvOnDev, simOnDev;

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

        cnvOnHost = HostBuffer<real>(145 * 45000);

        while (cur) {
            if (*cur != '\"')
              cnvOnHost[i++] = atof(cur);
            else ++nvec;
            cur = strtok(NULL, ",\n");
        }
        dim = i / nvec;
    }

    void normalizeDev() {
        int chunk = 8;
        vector<Stream> streams(nvec / chunk + 1);

        cnvOnDev = DevBuffer<real>(nvec * dim);
        for (int i = 0, off = 0; off < nvec; ++i, off += chunk) {
            int nbytes = min(nvec - off, chunk) * dim * sizeof(real);
            cudaMemcpyAsync(&cnvOnDev[off * dim], &cnvOnHost[off * dim], nbytes,
                          cudaMemcpyHostToDevice, streams[i]);
            
            device::normalize<256>
                <<<chunk, 256, 0, streams[i]>>>
                (cnvOnDev, off, nvec, dim);
        }
        cudaDeviceSynchronize();
    }

    void normalizeHost() {
        for (int i = 0; i < nvec; ++i) {
            real norm = 0;
            for (int j = 0; j < dim; ++j) {
                auto r = cnvOnHost[dim * i + j];
                norm += r * r;
            }
            norm = sqrt(norm);

            for (int j = 0; j < dim; ++j) {
                cnvOnHost[dim * i + j] /= norm;
            }
        }
    }

    void similarityDev() {
        dim3 block(8, 8);
        dim3 grid(nvec / block.x + 1, nvec / block.y + 1);
        simOnDev = DevBuffer<real>(nvec * nvec);
        device::similarity<<<grid, block>>>(cnvOnDev, dim, nvec, simOnDev);

        int nbytes = nvec * nvec * sizeof(real);
        simFromDev = HostBuffer<real>(nvec * nvec);
        verify(cudaMemcpy(simFromDev, simOnDev, nbytes, cudaMemcpyDeviceToHost));
    }

    static real scalarHost(real *x, real *y, int dim) {
        real total = 0;
        for (int i = 0; i < dim; ++i) {
            total += x[i] * y[i];
        }
        return total;
    }

    void similarityHost() {
        simOnHost = HostBuffer<real>(nvec * nvec);

        for (int i = 0; i < nvec; ++i) {
            real *x = cnvOnHost + dim * i;
            for (int j = i; j < nvec; ++j) {
                real *y = cnvOnHost + dim * j;
                simOnHost[i*nvec+j] = simOnHost[j*nvec+i] = scalarHost(x, y, dim);
            }
        }
    }

    static void statsHost(real *sim, int nvec) {
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

    static real correlationHost(real *x, real *y, int dim) {
        real xsum, ysum, xysum, xxsum, yysum;
        xsum = ysum = xysum = xxsum = yysum = 0;

        for (int id = 0; id < dim; ++id) {
            real xi = x[id], yi = y[id];
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
        fr = t.measure("Data");
        retrieveData();
        fr.resolve();

        fr = t.measure("Norm");
        normalizeDev();        
        fr.resolve();

        if (debugMode) {
            normalizeHost();

            cnvFromDev = HostBuffer<real>(nvec * dim);
            int nbytes = nvec * dim * sizeof(real);
            verify(cudaMemcpy(cnvFromDev, cnvOnDev, nbytes, cudaMemcpyDeviceToHost));

            auto corr = correlationHost(cnvFromDev, cnvOnHost, nvec * dim);
            cerr << "Correlation [CNVs]: " << corr << '\n';
        }

        fr = t.measure("Sim");
        similarityDev();
        fr.resolve();

        cerr << "Stats (simFromDev):\n"; 
        statsHost(simFromDev, nvec);

        if (debugMode) {
            similarityHost();
            cerr << "Stats (simOnHost):\n"; 
            statsHost(simOnHost, nvec);

            auto corr = correlationHost(simFromDev, simOnHost, nvec * nvec);
            cerr << "Correlation [Sims]: " << corr << '\n';
        }

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
        cout << "Exiting\n";
        return EXIT_FAILURE;
    }
}
