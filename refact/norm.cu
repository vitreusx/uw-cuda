#include "norm.h"

void host::norm(real *mat, int nvec, int dim) {
    for (int i = 0; i < nvec; ++i) {
        real norm = 0;
        for (int j = 0; j < dim; ++j) {
            auto r = mat[dim * i + j];
            norm += r * r;
        }

        norm = sqrt(norm);
        for (int j = 0; j < dim; ++j) {
            mat[dim * i + j] /= norm;
        }
    }
}
