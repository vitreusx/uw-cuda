TARGETS = sim trans query third
TARGETS_O = $(patsubst %,%.o,$(TARGETS))
all: $(TARGETS)

UTILS = cnv metrics timer utils stream
UTILS_O = $(patsubst %,utils/%.o,$(UTILS))

FLAGS = -O2 -std=c++11 -arch=sm_37 --ptxas-options=-v

%.o: %.cu
	nvcc $(FLAGS) -c -o $@ $^

%.o: %.cpp
	nvcc $(FLAGS) -c -o $@ $^

sim: sim.o $(UTILS_O)
	nvcc $(FLAGS) $^ -o $@

trans: trans.o $(UTILS_O)
	nvcc $(FLAGS) $^ -o $@ -lcurand

query: query.o
	nvcc $(FLAGS) $^ -o $@

third: third.o $(UTILS_O)
	nvcc $(FLAGS) $^ -o $@

.PHONY: clean
clean:
	rm -rf $(TARGETS) $(TARGETS_O) $(UTILS_O)

.PHONY: deliver
deliver:
	rsync -r . j.bednarz@cuda:~/exp
