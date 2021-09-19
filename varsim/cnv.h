#pragma once
#include <string>
#include "hostbuffer.h"
#include "defs.h"

class CNV {
public:
    int nvec, dim;
    HostBuffer<real> data;

    CNV() = default;
    CNV(std::string filename);
};
