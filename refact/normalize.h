#pragma once
#include "defs.h"
#include <cuda_runtime.h>

namespace host {
    void norm(real *mat, int nvec, int dim);
}

namespace dev {
    template<int Block>
    __global__ void norm_ker(real *mat, int nvec, int dim);
}

