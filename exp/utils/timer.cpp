#include "timer.h"
#include <iostream>
#include <cmath>
using namespace std;
using namespace std::chrono;

void Timer::enter(int N) {
    frames.emplace_back();
    auto& frame = frames.back();
    frame.N = N;

    auto now = high_resolution_clock::now();
    frame.prior = now;
}

void Timer::leave() {
    auto now = high_resolution_clock::now();

    auto& frame = frames.back();
    frame.after = now;
    
    duration<double> span = frame.after - frame.prior;
    frame.span = span.count();
}

Timer::Stats Timer::stats(bool clear) {
    Stats st;
    
    st.N = 0;
    st.total = 0;
    for (auto& frame: frames) {
        st.total += frame.span;
        st.N += frame.N;
    }

    if (st.N > 0) {
        st.mean = st.total / st.N;
        if (st.N > 1) {
            double sq = 0;
            for (auto& frame: frames) {
                double term = frame.span / (double)frame.N - st.mean;
                term *= term * frame.N;
                sq += term;
            }
            st.std = sqrt(sq / (st.N - 1));
        }
    }

    if (clear)
        this->clear();
    
    return st;
}

void Timer::clear() {
    frames = {};
}

std::ostream& operator<<(std::ostream& os, Timer::Stats const& st) {
    os << "N = " << st.N << "; Sum = " << st.total;
    if (st.N > 0) os << "; Avg = " << st.mean;
    if (st.N > 1) os << "; Std = " << st.std;
    return os;
}
