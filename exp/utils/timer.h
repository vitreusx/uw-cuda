#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <ostream>
#include "const.h"

class Timer {
private:
    struct Frame {
        using time_t = decltype(std::chrono::high_resolution_clock::now());
        time_t prior, after;
        int N;
        double span;
    };
    std::vector<Frame> frames;

public:
    struct Stats {
        int N;
        double total, mean, std;

        friend std::ostream& operator<<(std::ostream& os, Stats const& stats);
    };

    void enter(int N = 1);
    void leave();
    Stats stats(bool clear = true);
    void clear();
};
