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

// Default numeric type.
typedef float real;

bool debugMode = false;
#define WARP 32
#define TILE 32

namespace cuda {
// CUDA error-checking helper functions.
void check(cudaError_t ret) {
  if (ret != cudaSuccess) {
    throw runtime_error(cudaGetErrorString(ret));
  }
}

void check() { check(cudaGetLastError()); }

template <typename T> class Buffer {
private:
  T *ptr;

public:
  Buffer(size_t nitems) {
    cuda::check(cudaMalloc((void **)&ptr, sizeof(T) * nitems));
  }

  ~Buffer() { cuda::check(cudaFree(ptr)); }

  operator T *() { return ptr; }
};
} // namespace cuda

// Time measurement facility.
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

    void enter() { prior = clock(); }

    void leave() {
      time_t after = clock();
      real dur = (real)(after - prior) / CLOCKS_PER_SEC;
      times.push_back(dur);
    }

    void resolve(real &total, real &mean, real &std) {
      if (single)
        leave();

      if (debugMode)
        cerr << "[" << message << "] Measurement complete.\n";
      total = 0;
      for (size_t i = 0; i < times.size(); ++i) {
        total += times[i];
      }

      mean = total / times.size();
      std = 0;
      for (size_t i = 0; i < times.size(); ++i) {
        std += (times[i] - mean) * (times[i] - mean);
      }
      if (times.size() > 1)
        std /= times.size() - 1;
      std = sqrt(std);

      if (times.size() >= 1) {
        if (debugMode)
          cerr << "\t    N: " << times.size() << "\n"
               << "\tTotal: " << total << "s\n";
      }
      if (times.size() > 1) {
        if (debugMode)
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
    if (debugMode)
      cerr << "[" << msg << "] Measuring.\n";
    return Frame(msg, single);
  }
};

// Host-side numerical functions.
namespace host {
// Compute <x, y> for x, y: R^dim
real scalar(real *x, real *y, int dim) {
  real sum = 0;
  for (int i = 0; i < dim; ++i)
    sum += x[i] * y[i];
  return sum;
}

// Given {x_i} \subset R^dim of size nvec as src,
// output {norm(x_i)} to dest
void normalize(vector<real> &cnv, int dim, int nvec) {
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
      sim[i * nvec + j] = host::scalar(x, y, dim);
    }
  }
}

template <typename T> class Buffer {
private:
  T *ptr;

public:
  Buffer(size_t nitems) {
    if (!(ptr = (T *)malloc(sizeof(T) * nitems))) {
      throw std::runtime_error("malloc");
    }
  }

  ~Buffer() { free(ptr); }

  operator T *() { return ptr; }
};
} // namespace host

// Kernels &c.
namespace device {
template <int X, int Y>
__global__ void scalar_noshared(real *mat, int dim, int nvec, real *res) {
  // Skip lower echelon blocks (now in the standard fashion.)
  if (blockIdx.x * X > (blockIdx.y + 1) * Y)
    return;

  int xidx = blockIdx.x * X + threadIdx.x;
  if (xidx >= nvec)
    return;

  int yidx = blockIdx.y * Y + threadIdx.y;
  if (yidx >= nvec)
    return;

  real *x = &mat[dim * xidx], *y = &mat[dim * yidx];
  real total = 0;
  for (int i = 0; i < dim; ++i) {
    total += x[i] * y[i];
  }

  res[xidx * nvec + yidx] = total;
}

template <int X, int Y>
void similarity_noshared(real *cnv, real *sim, int dim, int nvec) {
  dim3 grid(nvec / X + 1, nvec / Y + 1);
  device::scalar_noshared<X, Y><<<grid, dim3(X, Y)>>>(cnv, dim, nvec, sim);
}

template <int X, int Y>
__global__ void scalar_conflicts(real *mat, int dim, int nvec, real *res) {
  if (blockIdx.x * X > (blockIdx.y + 1) * Y)
    return;

  int xidx = blockIdx.x * blockDim.x + threadIdx.x;
  int yidx = blockIdx.y * blockDim.y + threadIdx.y;
  int tidx = threadIdx.y * X + threadIdx.x;

  __shared__ real left[X][TILE], right[Y][TILE];
  real sum = 0.0f;
  const real *term = mat + dim * nvec;
  
  for (int i = 0; i < dim; i += TILE) {
    if (tidx < TILE && i + tidx < dim) {
      real *xvec = &mat[blockIdx.x * blockDim.x * dim + i + tidx];
      for (int x = 0; x < X && xvec < term; ++x, xvec += dim) {
        left[x][tidx] = *xvec;
      }

      real *yvec = &mat[blockIdx.y * blockDim.y * dim + i + tidx];
      for (int y = 0; y < Y && yvec < term; ++y, yvec += dim) {
        right[y][tidx] = *yvec;
      }
    }
    __syncthreads();

    for (int j = 0; j < TILE && i + j < dim; ++j) {
      sum += left[threadIdx.x][j] * right[threadIdx.y][j];
    }
    __syncthreads();
  }
  if (xidx < nvec && yidx < nvec)
    res[xidx * nvec + yidx] = sum;
}

template <int X, int Y>
void similarity_conflicts(real *cnv, real *sim, int dim, int nvec) {
  dim3 grid(nvec / X + 1, nvec / Y + 1);
  device::scalar_conflicts<X, Y><<<grid, dim3(X, Y)>>>(cnv, dim, nvec, sim);
}

template <int X, int Y>
__global__ void scalar_no_conflicts(real *mat, int dim, int nvec, real *res) {
  if (blockIdx.x * X > (blockIdx.y + 1) * Y)
    return;

  int xidx = blockIdx.x * blockDim.x + threadIdx.x;
  int yidx = blockIdx.y * blockDim.y + threadIdx.y;
  int tidx = threadIdx.y * X + threadIdx.x;

  __shared__ real left[X][TILE + 1], right[Y][TILE + 1];
  real sum = 0.0f;
  for (int i = 0; i < dim; i += TILE) {
    if (tidx < TILE && i + tidx < dim) {
      for (int x = 0, rx = blockIdx.x * blockDim.x; x < X && rx < nvec; ++x, ++rx) {
        left[x][tidx] = mat[rx * dim + i + tidx];
      }
      for (int y = 0, ry = blockIdx.y * blockDim.y; y < Y && ry < nvec; ++y, ++ry) {
        right[y][tidx] = mat[ry * dim + i + tidx];
      }
    }
    __syncthreads();

    for (int j = 0; j < TILE && i + j < dim; ++j) {
      sum += left[threadIdx.x][j] * right[threadIdx.y][j];
    }
    __syncthreads();
  }
  if (xidx < nvec && yidx < nvec)
    res[xidx * nvec + yidx] = sum;
}

template <int X, int Y>
void similarity_no_conflicts(real *cnv, real *sim, int dim, int nvec) {
  dim3 grid(nvec / X + 1, nvec / Y + 1);
  device::scalar_no_conflicts<X, Y><<<grid, dim3(X, Y)>>>(cnv, dim, nvec, sim);
}
} // namespace device

vector<real> readCSV(char *filename, int &dim, int &nvec) {
  size_t maxline = 2000000;
  host::Buffer<char> buf(maxline);

  // Reserve CNV.
  vector<real> cnv;
  cnv.reserve(dim * nvec);

  ifstream file(filename);
  size_t linum = 0;
  while (file.getline(buf, maxline)) {
    // Skip 1st line (i.e. the header)
    if (++linum > 1) {
      size_t colnum = 0;
      char *cur = strtok(buf, ",");
      while (cur) {
        // Skip 1st column; push values in flat form.
        if (++colnum > 1)
          cnv.push_back(atof(cur));
        cur = strtok(NULL, ",");
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
      real cur = sim[i * nvec + j] * sim[i * nvec + j];
      if (cur < min)
        min = cur;
      if (cur > max)
        max = cur;
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

int main(int argc, char **argv) {
  char *filename = NULL;
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
  Timer::Frame fr;

  fr = t.measure("Reading & parsing CSV file");
  int nvec = 150, dim = 50000;
  vector<real> cnv = readCSV(filename, dim, nvec);
  fr.resolve();

  fr = t.measure("Normalizing CNV");
  host::normalize(cnv, dim, nvec);
  fr.resolve();

  fr = t.measure("Allocating similarity matrices");
  host::Buffer<real> simHost(nvec * nvec);
  cuda::Buffer<real> simDev(nvec * nvec);
  host::Buffer<real> simDevLocal(nvec * nvec);
  fr.resolve();

  fr = t.measure("Allocating & copying CNV to the device");
  cuda::Buffer<real> cnvDev(nvec * dim);
  cuda::check(cudaMemcpy(cnvDev, cnv.data(), nvec * dim * sizeof(real),
                         cudaMemcpyHostToDevice));
  fr.resolve();

  if (debugMode) {
    fr = t.measure("Similarity (Host)");
    host::similarity(cnv.data(), simHost, dim, nvec);
    reportStats(simHost, nvec);
    fr.leave();
  }

  typedef void (*method_t)(real *, real *, int, int);
  method_t methods[] = {&device::similarity_noshared<8, 8>,
                        &device::similarity_conflicts<8, 8>,
                        &device::similarity_no_conflicts<8, 8>,

                        &device::similarity_noshared<16, 16>,
                        &device::similarity_conflicts<16, 16>,
                        &device::similarity_no_conflicts<16, 16>,

                        &device::similarity_noshared<24, 24>,
                        &device::similarity_conflicts<24, 24>,
                        &device::similarity_no_conflicts<24, 24>,

                        &device::similarity_noshared<32, 32>,
                        &device::similarity_conflicts<32, 32>,
                        &device::similarity_no_conflicts<32, 32>};
  string labels[] = {
      "NO SHARED <8, 8>",   "WITH CONFLICTS <8, 8>",   "NO CONFLICTS <8, 8>",
      "NO SHARED <16, 16>", "WITH CONFLICTS <16, 16>", "NO CONFLICTS <16, 16>",
      "NO SHARED <24, 24>", "WITH CONFLICTS <24, 24>", "NO CONFLICTS <24, 24>",
      "NO SHARED <32, 32>", "WITH CONFLICTS <32, 32>", "NO CONFLICTS <32, 32>"};

  for (int m = 0; m < sizeof(methods) / sizeof(method_t); ++m) {
    stringstream ss;
    ss << "Similarity (" << labels[m] << ")";

    fr = t.measure(ss.str(), false);
    for (size_t i = 0; i < 50; ++i) {
      fr.enter();
      methods[m](cnvDev, simDev, dim, nvec);
      if (debugMode)
        cuda::check();
      cuda::check(cudaMemcpy(simDevLocal, simDev, nvec * nvec * sizeof(real),
                             cudaMemcpyDeviceToHost));
      fr.leave();
    }
    real total, mean, std;
    fr.resolve(total, mean, std);

    cout << labels[m] << " TIME " << mean << '\n';

    if (debugMode) {
      reportStats(simDevLocal, nvec);
      real corr = correlation(simDevLocal, simHost, nvec);
      cerr << "Correlation: " << corr << '\n';
    }
  }

  return EXIT_SUCCESS;
}
