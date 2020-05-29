#include "stream.h"
#include "utils.h"

Stream::Stream() {
    check(cudaStreamCreate(&stream));
}

Stream::~Stream() {
    check(cudaStreamDestroy(stream));
    stream = 0;
}

Stream::operator cudaStream_t() {
    return stream;
}
