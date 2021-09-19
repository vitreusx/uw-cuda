#include "stream.cuh"
#include "utils.cuh"

Stream::Stream() {
    check(cudaStreamCreate(&stream));
}

Stream::~Stream() {
    check(cudaStreamDestroy(stream));
    stream = 0;
}
