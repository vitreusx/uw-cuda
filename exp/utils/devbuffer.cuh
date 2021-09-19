#pragma once
#include "utils.cuh"
#include <cstddef>
#include <cuda_runtime.h>

template <typename T>
class DevBuffer {
private:
    T *ptr;

public:
    DevBuffer() {
        ptr = NULL;
    }

    DevBuffer(int nelems) {
        check(cudaMalloc((void **)&ptr, sizeof(T) * nelems));
    }

    DevBuffer(DevBuffer const& other) = delete;
    DevBuffer& operator=(DevBuffer const& other) = delete;

    DevBuffer(DevBuffer&& other) {
        *this = move(other);
    }

    DevBuffer& operator=(DevBuffer&& other) {
        ptr = other.ptr;
        other.ptr = NULL;
        return *this;
    }

    ~DevBuffer() {
        if (ptr)
            check(cudaFree(ptr));
    }

    operator T *() {
        return ptr;
    }
};
