#pragma once

#include <cuda_runtime_api.h>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
using namespace std;

// Default numeric type.
typedef float real;

namespace host {
    // Compute <x, y> for x, y: R^dim
    real scalar(real *x, real *y, int dim) {
        real sum = 0;
        for (int i = 0; i < dim; ++i) sum += x[i] * y[i];
        return sum;
    }

    // Given {x_i} \subset R^dim of size nvec as src, output {norm(x_i)} to dest
    void normalize(real **src, real **dest, int dim, int nvec) {
        for (int i = 0; i < nvec; ++i) {
            real norm = sqrt(host::scalar(src[i], src[i], dim));
            for (int j = 0; j < dim; ++j) {
                dest[i][j] = src[i][j] / norm;
            }
        }
    }

    // Given {x_i} \subset R^dim of size nvec as normed_cnv, output a matrix \
(<x_i, x_j>)_{i,j}
    // in flattened form to sim
    void similarity(real **normed_cnv, real *sim, int dim, int nvec) {
        for (int i = 0; i < nvec; ++i) {
            for (int j = 0; j < nvec; ++j) {
                sim[i*nvec+j] = host::scalar(normed_cnv[i], normed_cnv[j], dim);
            }
        }
    }
}

namespace device {
    // Compute <x, y> for x, y: R^dim and output into *res
    // Appropriate implementations are placed in redukcjaN.cu
    // !Important! Assumes there is precisely 1 block.
    template<size_t Block>
    __global__ void scalar(real *x, real *y, int dim, real *res);

    // (Same as on host)
    void similarity(real *normed_cnv, real *sim, int dim, int nvec) {
        for (int i = 0; i < nvec; ++i) {
            real *x = normed_cnv + i * dim;
            for (int j = 0; j < nvec; ++j) {
                real *y = normed_cnv + j * dim;
                device::scalar<1024><<<1, 1024, 0>>>(x, y, dim, &sim[i * nvec + j]);
            }
        }
    }
}

// Parse provided csv file and output as a matrix into CNV
// return dim of resultant vectors
int read_csv(char* CSVfile,real **CNV) {
    FILE* CNVfile;
    real* row;
    const int buf_size=2000000;
    const int max_cols=100000;
    char buffer[buf_size];
    int col_count;

    row = (float*) malloc(max_cols*sizeof(float));
    CNVfile = fopen(CSVfile,"r");
    int line_count =0;

    int row_count;
    while (fgets(buffer, 1999999, CNVfile)){
        line_count++;
        if (line_count>1){
            row_count = line_count-2;
            col_count=-1;
            char *col = strtok(buffer, ",");
            while (col) {
                if (col_count >= 0) {
                    row[col_count]=atof(col);
                }
                col = strtok(NULL, ",");
                col_count++;
            }
            CNV[row_count]= (float*) malloc((col_count+1)*sizeof(float));
            for (int i=0;i<=col_count;i++) CNV[row_count][i]=row[i];
        }
    }

    fclose(CNVfile);
    return col_count;
}

// Compute avg, min and max of similarity matrix sim: R^{nvec,nvec}
void report_stats(real *sim, int nvec) {
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
    cout << "Minimum similarity (%): " << 100 * min << '\n';
    cout << "Maximum similarity (%): " << 100 * max << '\n';
    cout << "Average similarity (%): " << 100 * total << '\n';
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

class timer {
private:
    time_t prior;
    string message;

public:
    void start(string msg) {
        this->message = msg;
        prior = clock();
        cout << "[" << this->message << "] Starting timer." << '\n';
    }

    real end() {
        int after = clock();
        real dur = (real)(after - prior) / CLOCKS_PER_SEC;
        cout << "[" << this->message << "] Timing done: took " << dur << "s." << '\n';
        return dur;
    }
};

int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "Wrong number of arguments\n";
        cout << "Usage: " << argv[0] << " filename\n";
        cout << "Exiting\n";
        return EXIT_FAILURE;
    }

    // Load data into an array.
    const int nvec = 145;
    real *CNV[nvec];
    int dim = read_csv(argv[1], CNV);

    // Normalize x_i's (host-side).
    real *normCNV[nvec];
    for (int i = 0; i < nvec; ++i)
        normCNV[i] = (real *)malloc(dim * sizeof(real));
    host::normalize(CNV, normCNV, dim, nvec);

    timer t;
    cudaError_t ret;

    t.start("Flattening normCNV");
    real *flatNormCNV = (real *)malloc(dim*nvec * sizeof(real));
    if (!flatNormCNV)
        throw runtime_error("malloc(flatNormCNV)");
    for (int i = 0; i < nvec; ++i) {
        memcpy(flatNormCNV + i*dim, normCNV[i], dim * sizeof(real));
    }
    t.end();

    // Create similarity matrices.
    real *simHost = (real *)malloc(nvec*nvec * sizeof(real));
    if (!simHost)
        throw runtime_error("malloc(simHost)");

    real *simDev;
    ret = cudaMalloc((void **)&simDev, nvec*nvec * sizeof(real));
    if (ret != cudaSuccess)
        throw runtime_error("cudaMalloc(simDev)");

    real *simDevLocal = (real *)malloc(nvec*nvec * sizeof(real));
    if (!simDevLocal)
        throw runtime_error("malloc(simDevLocal)");

    // Create & copy flat normCNV over to the device.
    real *cnvDev;
    ret = cudaMalloc((void **)&cnvDev, nvec*dim * sizeof(real));
    if (ret != cudaSuccess)
        throw runtime_error("cudaMalloc(cnvDev)");
    ret = cudaMemcpy(cnvDev, flatNormCNV, nvec*dim * sizeof(real), cudaMemcpyHostToDevice);
    if (ret != cudaSuccess)
        throw runtime_error("cudaMemcpy(cnvDev, flatNormCNV)");

    t.start("Computing similarity (Host)");
    host::similarity(normCNV, simHost, dim, nvec);
    real timeCPU = t.end();
    report_stats(simHost, nvec);

    t.start("Computing similarity (Device)");
    device::similarity(cnvDev, simDev, dim, nvec);
    ret = cudaMemcpy(simDevLocal, simDev, nvec*nvec * sizeof(real), cudaMemcpyDeviceToHost);
    if (ret != cudaSuccess)
        throw runtime_error("cudaMemcpy(simDevLocal, simDev)");
    real timeGPU = t.end();
    report_stats(simDevLocal, nvec);

    real corr = correlation(simHost, simDevLocal, nvec*nvec);
    cout << "Correlation: " << corr << '\n';

    cout << "Results in requested format:" << '\n';
    fprintf(stdout, "KORELACJA %10.6f\n", corr);
    fprintf(stdout, "CZAS CPU %10.6f GPU %10.6f\n", timeCPU, timeGPU);

    return EXIT_SUCCESS;
}
