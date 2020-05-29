#pragma once
#include "utils.h"
#include <cstddef>
#include <cuda_runtime.h>

template <typename T>
class HostBuffer {
private:
    T *ptr;

public:
    HostBuffer() {
        ptr = NULL;
    }

    HostBuffer(int nelems) {
        check(cudaMallocHost((void **)&ptr, sizeof(T) * nelems));
    }

    HostBuffer(HostBuffer const& other) = delete;
    HostBuffer& operator=(HostBuffer const& other) = delete;

    HostBuffer(HostBuffer&& other) {
        *this = move(other);
    }

    HostBuffer& operator=(HostBuffer&& other) {
        ptr = other.ptr;
        other.ptr = NULL;
        return *this;
    }

    ~HostBuffer() {
        if (ptr)
            check(cudaFreeHost(ptr));
    }

    operator T *() {
        return ptr;
    }
};