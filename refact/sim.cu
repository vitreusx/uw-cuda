#include "sim.h"

static real dot(real *x, real *y, int dim) {
    real s = 0;
    for (int i = 0; i < dim; ++i, ++x, ++y) {
        s += *x * *y;
    }
    return s;
}

void host::sim(real *mat, int nvec, int dim, real *sim) {
    real *x = mat;
    for (int i = 0; i < nvec; ++i, x += dim) {
        real *y = x;
        for (int j = i; j < nvec; ++j, y += dim) {
            sim[i*nvec+j] = sim[j*nvec+i] = dot(x, y, dim);
        }
    }
}
