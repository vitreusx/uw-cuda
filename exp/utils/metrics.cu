#include "metrics.cuh"
#include "scalar.cuh"
#include "devbuffer.cuh"
#include "stream.cuh"
#include <iostream>
#include <cmath>
using namespace std;

ostream& operator<<(ostream& os, Stats const& st) {
    os << "Avg " << 100 * st.avg << "%; ";
    os << "Min " << 100 * st.min << "%; ";
    os << "Max " << 100 * st.max << "%";
    return os;
}

Stats host::stats(real *sim, int nvec) {
    Stats s;
    s.avg = 0;
    s.max = 0;
    s.min = 1;

    for (int i = 0; i < nvec; ++i) {
        for (int j = i + 1; j < nvec; ++j) {
            real cur = sim[i*nvec+j] * sim[i*nvec+j];
            if (cur < s.min) s.min = cur;
            if (cur > s.max) s.max = cur;
            s.avg += cur;
        }
    }

    s.avg /= (nvec * (nvec - 1)) / 2;
    return s;
}

real host::corr(real *x, real *y, int dim) {
    real xsum, ysum, xysum, xxsum, yysum;
    xsum = ysum = xysum = xxsum = yysum = 0;

    for (int id = 0; id < dim; ++id) {
        real xi = x[id], yi = y[id];
        xsum += xi;
        ysum += yi;
        xysum += xi * yi;
        xxsum += xi * xi;
        yysum += yi * yi;
    }

    real xmean = xsum / dim, ymean = ysum / dim;
    real covxy = xysum - dim * xmean * ymean;
    real varx = xxsum - dim * xmean * xmean;
    real vary = yysum - dim * ymean * ymean;
    return covxy / sqrt(varx * vary);
}

real dev::corr(real *x, real *y, int dim) {
    DevBuffer<real> vars(5);
    Stream streams[5];
    real hvars[5];

    dev::sum<1024><<<1, 1024, 0, streams[0]>>>(x, dim, &vars[0]);
    dev::sum<1024><<<1, 1024, 0, streams[1]>>>(y, dim, &vars[1]);
    dev::scalar<1024><<<1, 1024, 0, streams[2]>>>(x, y, dim, &vars[2]);
    dev::scalar<1024><<<1, 1024, 0, streams[3]>>>(x, x, dim, &vars[3]);
    dev::scalar<1024><<<1, 1024, 0, streams[4]>>>(y, y, dim, &vars[4]);
    check(cudaDeviceSynchronize());
    check(cudaMemcpy(hvars, vars, sizeof(hvars), cudaMemcpyDeviceToHost));

    real xsum = hvars[0], ysum = hvars[1],
         xysum = hvars[2], xxsum = hvars[3], yysum = hvars[4];
    
    real xmean = xsum / dim, ymean = ysum / dim;
    real covxy = xysum - dim * xmean * ymean;
    real varx = xxsum - dim * xmean * xmean;
    real vary = yysum - dim * ymean * ymean;
    return covxy / sqrt(varx * vary);
}
