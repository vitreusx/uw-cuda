#include "cnv.h"
#include <fstream>
#include <cstring>
using namespace std;

CNV::CNV(string filename) {
    ifstream file(filename.c_str());

    file.seekg(0, ios::end);
    int nbytes = file.tellg();
    file.seekg(0, ios::beg);

    HostBuffer<char> filebuf(nbytes);
    file.read(filebuf, nbytes);

    char *cur = filebuf;
    while (*cur != '\n') ++cur;
    cur = strtok(cur, ",\n");
        
    dim = nvec = 0;
    int i = 0;

    data = HostBuffer<real>(145 * 40000);
    while (cur) {
        if (*cur != '\"') data[i++] = atof(cur);
        else ++nvec;
        cur = strtok(NULL, ",\n");
    }
    dim = i / nvec;
}
