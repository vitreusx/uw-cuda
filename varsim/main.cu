#include "cnv.h"
#include "devbuffer.h"
#include "hostbuffer.h"
#include "norm.h"
#include "sim.h"
#include "stream.h"
#include "timer.h"
#include "utils.h"
#include "metrics.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace std;

class Main {
private:
    CNV cnv;
    DevBuffer<real> cnvOnDev, simOnDev;
    HostBuffer<real> simFromDev, cnvFromDev, simOnHost;

    Timer t;
    Timer::Frame fr;

    string filename;
    bool debugMode;

    void copyToDev(int numb, int off, cudaStream_t str = 0) {
        real *dst = cnvOnDev + off * cnv.dim;
        real *src = cnv.data + off * cnv.dim;
        int nbytes = sizeof(real) * cnv.dim * numb;

        check(cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyHostToDevice, str));
    }

    void norm(int numb, int off, cudaStream_t str = 0) {
        dim3 grid(numb);
        dim3 block(256);
        real *dst = cnvOnDev + off * cnv.dim;
        dev::norm_ker<256><<<grid, block, 0, str>>>(dst, cnv.dim);
        check();
    }

    void sim(dim3 ext, dim3 off, cudaStream_t str = 0) {
        dim3 block(16, 16);
        dim3 grid(ext.x / block.x + 1, ext.y / block.y + 1);

        dev::sim_ker<16, 16><<<grid, block, 0, str>>>(cnvOnDev, ext, off, cnv.nvec, cnv.dim, simOnDev);
        check();
    }

    void copySim() {
        int nbytes = sizeof(real) * cnv.nvec * cnv.nvec;
        check(cudaMemcpy(simFromDev, simOnDev, nbytes, cudaMemcpyDeviceToHost));
    }

    void overlapped() {
        int part1 = cnv.nvec - cnv.nvec / 2, part2 = cnv.nvec / 2;
        vector<Stream> streams(3);

        cnvOnDev = DevBuffer<real>(cnv.dim * cnv.nvec);
        simOnDev = DevBuffer<real>(cnv.nvec * cnv.nvec);
        
        copyToDev(part1, 0, streams[0]);
        norm(part1, 0, streams[0]);
        sim(dim3(part1, part1), dim3(0, 0), streams[0]);

        copyToDev(part2, part1, streams[1]);
        norm(part2, part1, streams[1]);
        cudaDeviceSynchronize();

        sim(dim3(part1, part2), dim3(0, part1), streams[0]);
        sim(dim3(part2, part2), dim3(part1, part1), streams[1]);
        sim(dim3(part2, part1), dim3(part1, 0), streams[2]);
        cudaDeviceSynchronize();

        simFromDev = HostBuffer<real>(cnv.nvec * cnv.nvec);
        copySim();    
    }

    void nonOverlapped() {
        int chunk = 16;
        vector<Stream> streams(cnv.nvec / chunk + 1);

        cnvOnDev = DevBuffer<real>(cnv.dim * cnv.nvec);
        simOnDev = DevBuffer<real>(cnv.nvec * cnv.nvec);

        for (int off = 0, ns = 0; off < cnv.nvec; off += chunk, ++ns) {
            int span = min(chunk, cnv.nvec - off);
            copyToDev(span, off, streams[ns]);
            norm(span, off, streams[ns]);
        }
        cudaDeviceSynchronize();

        sim(dim3(cnv.nvec, cnv.nvec), dim3(0, 0));

        simFromDev = HostBuffer<real>(cnv.nvec * cnv.nvec);
        copySim();
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

        fr = t.measure("Over", false);
        for (int i = 0; i < 50; ++i) {
            fr.enter();
            overlapped();
            fr.leave();
        }
        fr.resolve();

        fr = t.measure("Non-Over", false);
        for (int i = 0; i < 50; ++i) {
            fr.enter();
            nonOverlapped();
            fr.leave();
        }
        fr.resolve();

        if (debugMode) {
            host::norm(cnv.data, cnv.nvec, cnv.dim);

            cnvFromDev = HostBuffer<real>(cnv.nvec * cnv.dim);
            int nbytes = sizeof(real) * cnv.nvec * cnv.dim;
            check(cudaMemcpy(cnvFromDev, cnvOnDev, nbytes, cudaMemcpyDeviceToHost));

            auto c = host::corr(cnvFromDev, cnv.data, cnv.nvec * cnv.dim);
            cerr << "Corr [CNV]: " << c << '\n';
            
            cerr << "Stats (simFromDev):\n";
            host::stats(simFromDev, cnv.nvec).print();

            cerr << "Stats (simOnHost):\n";
            host::norm(cnv.data, cnv.nvec, cnv.dim);
            simOnHost = HostBuffer<real>(cnv.nvec * cnv.nvec);
            host::sim(cnv.data, cnv.nvec, cnv.dim, simOnHost);
            host::stats(simOnHost, cnv.nvec).print();

            c = host::corr(simFromDev, simOnHost, cnv.nvec * cnv.nvec);
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
