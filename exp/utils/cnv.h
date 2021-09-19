#pragma once
#include <string>
#include "hostbuffer.cuh"
#include "const.h"

class CNV {
public:
    int nvec, dim;
    HostBuffer<real> data;

    CNV() = default;
    CNV(std::string filename);
};
