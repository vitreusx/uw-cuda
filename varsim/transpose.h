#pragma once
#include "defs.h"
#include <cuda_runtime.h>

namespace host {
    void transpose(real *src, real *dst, dim3 sh, dim3 ext, dim3 off);
}

namespace dev {
    __device__ void tr_shared_ker(real *src, real *dst, dim3 sh, dim3 ext, dim3 off);
    __device__ void tr_reg_ker(real *src, real *dst, dim3 sh, dim3 ext, dim3 off);
}
