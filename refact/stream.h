#pragma once
#include <cuda_runtime.h>

class Stream {
private:
    cudaStream_t stream;

public:
    Stream();
    ~Stream();

    inline operator cudaStream_t();
};
