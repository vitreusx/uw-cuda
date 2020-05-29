#pragma once
#include <string>
#include <vector>
#include "defs.h"

class Timer {
public:
    class Frame {
    private:
        time_t prior;
        std::string message;
        std::vector<real> times;
        bool single;

        struct Results {
            real total;
            real mean;
            real std;
        };

    public:
        Frame() = default;
        Frame(std::string msg, bool _single);

        void enter();
        void leave();

        Results resolve();
    };

    Frame measure(std::string msg, bool single = true);
};
