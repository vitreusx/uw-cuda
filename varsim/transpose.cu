#include "transpose.h"
#include <cstring>

void host::transpose(real *src, real *dst, dim3 sh, dim3 ext, dim3 off) {
    for (int i = off.x; i < off.x + ext.x; ++i) {
        for (int j = off.y; j < off.y + ext.y; ++j) {
            dst[j * sh.x + i] = src[i * sh.y + j];        
        }
    }
}

__device__ void dev::tr_shared_ker(real *src, real *dst, dim3 sh, dim3 ext, dim3 off) {
    
}

__device__ void dev::tr_reg_ker(real *src, real *dst, dim3 sh, dim3 ext, dim3 off) {
    
}