#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <locale>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <algorithm>
#include <sstream>
using namespace std;
using namespace std::chrono;

using real = float;
using cuda_ptr = unique_ptr<real, void (*)(void*)>;
constexpr size_t Block = 512;
constexpr size_t Block2D = 16;

class csv_ctype : public ctype<char> {
    mask table[table_size];

public:
    csv_ctype(size_t refs = 0) : ctype<char>(&table[0], false, refs) {
        copy_n(classic_table(), table_size, table);
        table[','] = (mask)space;
    }
};

string load_file(ifstream&& src) {
    string buf;

    src.seekg(0, ios::end);
    buf.resize(src.tellg());
    src.seekg(0, ios::beg);
    src.read(&buf[0], buf.size());

    return buf;
}

pair<size_t, vector<real>> read_from_CSV(ifstream&& src) {
    string contents = load_file(move(src));
    stringstream ss {move(contents)};
    locale csv_locale { locale::classic(), new csv_ctype };
    ss.imbue(csv_locale);

    string header;
    getline(ss, header);
    auto nfactors = count(header.begin(), header.end(), ',');

    vector<real> cases;
    constexpr auto max_ncases = 150;
    cases.reserve(max_ncases * nfactors);

    while (true) {
        string id;
        if (!(ss >> id)) break;

        for (size_t i = 0; i < nfactors; ++i) {
            float column;
            ss >> column;
            cases.push_back(column);
        }
    }

    return make_pair(nfactors, cases);
}

class CUDAError : public exception {
private:
    cudaError_t retcode;

public:
    CUDAError(cudaError_t retcode) :
        retcode{ retcode } {};

    char const* what() const noexcept override {
        return cudaGetErrorString(retcode);
    }
};

void cudaCheck(cudaError_t retcode) {
    if (retcode != cudaSuccess)
        throw CUDAError{ retcode };
}

template<typename T>
T* cudaMalloc_Redux(size_t nbytes) {
    T* res;
    cudaCheck(cudaMalloc(&res, nbytes));
    return res;
}

void cudaFree_Redux(void* ptr) {
    cudaCheck(cudaFree(ptr));
}

struct level {
    dim3 grid;
    cuda_ptr mem;

    level(dim3 _grid, cuda_ptr &&_mem) :
        grid {_grid}, mem {move(_mem)} {};
};

vector<level> derive_levels(size_t x) {
    vector<level> res;
    while (x > 1) {
        cuda_ptr mem{
            cudaMalloc_Redux<real>(x * sizeof(real)),
            cudaFree_Redux
        };
        res.emplace_back(dim3(x, 1, 1), move(mem));
        x = (x - 1) / Block + 1;
    }
    return res;
}

__global__ void load_prods(real* u, real* v, real* out, size_t len) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) out[idx] = u[idx] * v[idx];
}

__global__ void reduce_scalar(real* in, real* out, size_t len) {
    __shared__ real auxl[Block / 2];

    size_t tidx = threadIdx.x;
    size_t idx = blockIdx.x * (2 * blockDim.x) + tidx;

    auxl[tidx] = 0;
    if (idx < len) auxl[tidx] += in[idx];
    if (idx + blockDim.x < len) auxl[tidx] += in[idx + blockDim.x];
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > tidx; s >>= 1) {
        auxl[tidx] += auxl[tidx + s];
        __syncthreads();
    }

    if (tidx == 0) out[blockIdx.x] = auxl[tidx];
}

void scalar(real* u, real* v, size_t len, real* res, vector<level>& levels) {
    dim3 load_block(Block, 1, 1);
    dim3 load_grid((len - 1) / load_block.x + 1, 1, 1);
    load_prods<<<load_grid, load_block, 0>>>(u, v, levels[0].mem.get(), len);

    dim3 block(Block / 2, 1, 1);
    for (size_t i = 0; i < levels.size(); ++i) {
        auto* in = levels[i].mem.get();
        auto* out = i + 1 < levels.size() ? levels[i + 1].mem.get() : res;
        auto len = levels[i].grid.x;

        dim3 grid((levels[i].grid.x - 1) / Block + 1, 1, 1);
        reduce_scalar<<<grid, block, 0>>>(in, out, len);
    }
}

__global__ void load_norms(real* simil, real* out, size_t len) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < len) out[idx] = sqrt(simil[idx * len + idx]);
}

__global__ void divide(real* simil, real* norms, size_t len) {
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < len && y < len) simil[y * len + x] /= norms[x] * norms[y];
}

void normalise(real* simil, size_t len) {
    cuda_ptr norms {
        cudaMalloc_Redux<real>(len * sizeof(real)),
        cudaFree_Redux
    };

    dim3 load_block(Block, 1, 1);
    size_t grid1d_len = (len - 1) / Block + 1;
    dim3 load_grid(grid1d_len, 1, 1);
    load_norms<<<load_grid, load_block, 0>>>(simil, norms.get(), len);

    dim3 div_block(Block2D, Block2D, 1);
    size_t grid2d_len = (len - 1) / Block2D + 1;
    dim3 div_grid(grid2d_len, grid2d_len, 1);
    divide<<<div_grid, div_block, 0>>>(simil, norms.get(), len);
}

class timer {
private:
    using time_point = decltype(high_resolution_clock::now());
    time_point prior, after;

public:
    void enter(string message) {
        cout << message << "...";
        cout.flush();
        prior = high_resolution_clock::now();
    }

    void leave() {
        after = high_resolution_clock::now();
        auto dur = (duration<real>(after - prior)).count();
        cout << " done: took " << dur << "s\n";
    }
};

int main() {
    timer t;

    t.enter("Loading CSV");
    auto data = read_from_CSV(ifstream("neuroblastoma_CNV.csv"));
    t.leave();

    auto& nfactors = data.first;
    auto& cases = data.second;
    auto ncases = cases.size() / nfactors;

    cuda_ptr cases_GPU {
        cudaMalloc_Redux<real>(cases.size() * sizeof(real)),
        cudaFree_Redux
    };
    cudaMemcpy(cases_GPU.get(), cases.data(), cases.size() * sizeof(real), cudaMemcpyHostToDevice);

    auto levels = derive_levels(nfactors);

    cuda_ptr simil_GPU {
        cudaMalloc_Redux<real>(ncases * ncases * sizeof(real)),
        cudaFree_Redux
    };

    t.enter("Computing <u_i, u_j> forall i, j");
    for (size_t i = 0; i < ncases; ++i) {
        auto* u = cases_GPU.get() + i * nfactors;
        for (size_t j = 0; j < ncases; ++j) {
            auto* v = cases_GPU.get() + j * nfactors;
            auto* res = simil_GPU.get() + i * ncases + j;
            scalar(u, v, nfactors, res++, levels);
        }
    }
    t.leave();

    normalise(simil_GPU.get(), ncases);

    vector<real> simil_CPU;
    simil_CPU.reserve(ncases * ncases);
    cudaMemcpy(simil_CPU.data(), simil_GPU.get(), ncases * ncases * sizeof(real), cudaMemcpyDeviceToHost);

    real simil_min = 2.0f, simil_max = -1.0f, simil_avg = 0.0f;
    for (size_t i = 0; i < ncases; ++i) {
        for (size_t j = 0; j < ncases; ++j) {
            if (simil_min > fabs(simil_CPU[i * ncases + j]))
                simil_min = fabs(simil_CPU[i * ncases + j]);
            if (i != j && simil_max < simil_CPU[i * ncases + j])
                simil_max = simil_CPU[i * ncases + j];
            if (i != j)
                simil_avg += simil_CPU[i * ncases + j];
        }
    }
    simil_avg /= ncases * (ncases - 1);

    cout << "min |<u_i, u_j>| = " << simil_min << '\n';
    cout << "max_{i != j} <u_i, u_j> = " << simil_max << '\n';
    cout << "avg_{i != j} <u_i, u_j> = " << simil_avg << '\n';

    for (size_t i = 0; i < ncases; ++i) {
        cout << "simil[" << i << "][" << i << "] = " << simil_CPU[i * ncases + i] << '\n';
    }

    return EXIT_SUCCESS;
}
