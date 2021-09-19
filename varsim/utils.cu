#include "utils.h"
#include <stdexcept>
using namespace std;

void check(cudaError_t ret) {
    if (ret != cudaSuccess) {
        throw runtime_error(cudaGetErrorString(ret));
    }
}

void check() {
    check(cudaGetLastError());
}
