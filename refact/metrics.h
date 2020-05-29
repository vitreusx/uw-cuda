#pragma once
#include "defs.h"

struct Stats {
    real min;
    real max;
    real avg;

    void print();
};

namespace host {
    Stats stats(real *sim, int nvec);
    real corr(real *x, real *y, int dim);
}
