#include "cnv.h"
#include "devbuffer.h"
#include "hostbuffer.h"
#include "normalize.h"
#include "sim.h"
#include "stream.h"
#include "timer.h"
#include "utils.h"
#include "metrics.h"

#include <iostream>
#include <stdexcept>
#include <string>
using namespace std;

class Main {
private:
    CNV cnv;
    DevBuffer<real> cnvOnDev, simOnDev;
    HostBuffer<real> simFromDev, simOnHost;

    Timer t;
    Timer::Frame fr;

    string filename;
    bool debugMode;

    void normDev() {
        host::norm(cnv.data, cnv.nvec, cnv.dim);

        cnvOnDev = DevBuffer<real>(cnv.nvec * cnv.dim);
        int nbytes = sizeof(real) * cnv.nvec * cnv.dim;
        check(cudaMemcpy(cnvOnDev, cnv.data, nbytes, cudaMemcpyHostToDevice));
    }

    void simDev() {
        simOnDev = DevBuffer<real>(cnv.nvec * cnv.nvec);
        dim3 block(16, 16);
        dim3 grid(cnv.nvec / block.x + 1, cnv.nvec / block.y + 1);
        dev::sim_ker<16, 16><<<grid, block>>>(cnvOnDev, cnv.nvec, cnv.dim, simOnDev);

        int nbytes = sizeof(real) * cnv.nvec * cnv.nvec;
        simFromDev = HostBuffer<real>(cnv.nvec * cnv.nvec);
        check(cudaMemcpy(simFromDev, simOnDev, nbytes, cudaMemcpyDeviceToHost));
    }

public:
    Main(int argc, char **argv) {
        char *filenamePtr = NULL;
        debugMode = false;

        for (int i = 1; i < argc; ++i) {
            if (!debugMode && (!strcmp(argv[i], "-g") || !strcmp(argv[i], "--debug"))) 
                debugMode = true;
            else if (!filenamePtr)
                filenamePtr = argv[i];
        }

        if (!filenamePtr) throw invalid_argument("argv");
        else filename = filenamePtr;

        if (!debugMode)
            cerr.setstate(ios::failbit);
    }

    int run() {        
        fr = t.measure("CNV");
        cnv = CNV(filename.c_str());
        fr.resolve();

        fr = t.measure("Norm");
        normDev();
        fr.resolve();

        fr = t.measure("Sim");
        simDev();
        fr.resolve();

        if (debugMode) {
            cerr << "Stats (simFromDev):\n";
            host::stats(simFromDev, cnv.nvec).print();

            cerr << "Stats (simOnHost):\n";
            simOnHost = HostBuffer<real>(cnv.nvec * cnv.nvec);
            host::sim(cnv.data, cnv.nvec, cnv.dim, simOnHost);
            host::stats(simOnHost, cnv.nvec).print();

            auto c = host::corr(simFromDev, simOnHost, cnv.nvec * cnv.nvec);
            cerr << "Corr [Sim]: " << c << '\n';
        }

        return EXIT_SUCCESS;
    }
};

int main(int argc, char **argv) {
    try {
        return Main(argc, argv).run();
    }
    catch (invalid_argument) {
        cout << "Wrong number of arguments\n";
        cout << "Usage: " << argv[0] << " [-g|--debug] filename\n";
        cout << "Exiting\n";
        return EXIT_FAILURE;
    } 
    catch (exception &e) {
        cerr.clear();
        cerr << "[ERROR] Message: " << e.what() << endl;
        cout << "Exiting\n";
        return EXIT_FAILURE;
    }
}