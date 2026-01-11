HIPCC ?= /opt/rocm/bin/hipcc
# HIPFLAGS ?= -O3 -DNDEBUG --offload-arch=native
HIPFLAGS ?= -O3 --offload-arch=native

TARGETS := test_dd_f32 test_dd_f32_full some_complex_calculation

.PHONY: all clean run

all: $(TARGETS)

test_dd_f32: test_dd_f32.hip dd_f32.h
	$(HIPCC) $(HIPFLAGS) $< -o $@

test_dd_f32_full: test_dd_f32_full.hip dd_f32.h
	$(HIPCC) $(HIPFLAGS) $< -o $@

some_complex_calculation: some_complex_calculation.hip dd_f32.h
	$(HIPCC) $(HIPFLAGS) $< -o $@

run: all
	./test_dd_f32
	./test_dd_f32_full
	./some_complex_calculation

clean:
	rm -f $(TARGETS) *.o
