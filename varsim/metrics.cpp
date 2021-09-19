#include "metrics.h"
#include <iostream>
#include <cmath>
using namespace std;

void Stats::print() {
    cerr << "Avg: " << 100 * avg << "%\n";
    cerr << "Min: " << 100 * min << "%\n";
    cerr << "Max: " << 100 * max << "%\n";
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
