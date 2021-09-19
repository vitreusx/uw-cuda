SOURCES = utils cnv metrics norm \
          sim stream timer main \
		  transpose

%.o: %.cu
	nvcc -O2 -std=c++11 -c -o $@ $^

%.o: %.cpp
	nvcc -O2 -std=c++11 -c -o $@ $^

simvar: $(patsubst %,%.o,$(SOURCES))
	nvcc -O2 -std=c++11 $^ -o $@

.PHONY: clean
clean:
	rm -rf $(patsubst %,%.o,$(SOURCES)) refact

.PHONY: deliver
deliver:
	rsync -r . j.bednarz@cuda:~/varsim
