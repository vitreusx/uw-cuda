#include "timer.h"
#include <iostream>
#include <cmath>
using namespace std;

Timer::Frame::Frame(string msg, bool _single) {
    message = msg;
    single = _single;
    if (single)
        enter();
}

void Timer::Frame::enter() {
    prior = clock();
}

void Timer::Frame::leave() {
    time_t after = clock();
    real dur = (real)(after - prior) / CLOCKS_PER_SEC;
    times.push_back(dur);
}

Timer::Frame::Results Timer::Frame::resolve() {
    Results res;

    if (single)
        leave();

    cerr << "[" << message << "] Measurement complete.\n";
    res.total = 0;
    for (int i = 0; i < times.size(); ++i) {
        res.total += times[i];
    }

    res.mean = res.total / times.size();
    res.std = 0;
    for (int i = 0; i < times.size(); ++i) {
        res.std += (times[i] - res.mean) * (times[i] - res.mean);
    }
    if (times.size() > 1)
        res.std /= times.size() - 1;
    res.std = sqrt(res.std);

    if (times.size() >= 1) {
    cerr << "\t    N: " << times.size() << "\n"
         << "\tTotal: " << res.total << "s\n";
    }
    if (times.size() > 1) {
        cerr << "\t Mean: " << res.mean << "s\n"
             << "\t  Std: " << res.std << "s\n";
    }

    return res;
}

Timer::Frame Timer::measure(string msg, bool single) {
    cerr << "[" << msg << "] Measuring.\n";
    return Frame(msg, single);
}
