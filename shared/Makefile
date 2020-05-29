TARGETS: shared sharedx sharedy

shared: shared.cu
	nvcc -o $@ -O2 $^

sharedx: sharedx.cu
	nvcc -o $@ -O2 $^

sharedy: sharedy.cu
	nvcc -o $@ -O2 $^

.PHONY: clean
clean:
	rm -rf $(TARGETS)
