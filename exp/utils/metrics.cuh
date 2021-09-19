#pragma once
#include "const.h"
#include <ostream>

struct Stats {
    real min;
    real max;
    real avg;

    friend std::ostream& operator<<(std::ostream& os, Stats const& st);
};

namespace host {
    Stats stats(real *sim, int nvec);
    real corr(real *x, real *y, int dim);
}

namespace dev {
    real corr(real *x, real *y, int dim);
}
