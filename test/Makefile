CXX=mpicxx
DEFS=
INCLUDES=-I$(WORK2)/../cpr-perf-model/include/
LDFLAGS=-L$(WORK2)/../cpr-perf-model/lib/
LIBS=-lcpr_perf_model
CXXFLAGS=-g -O3 -fopenmp $(DEFS) -std=c++0x $(INCLUDES) $(LDFLAGS) $(LIBS) -L/opt/intel/compilers_and_libraries_2018.2.199/linux/mkl/lib/intel64/ -lmkl_scalapack_lp64 -Wl,--start-group -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lmkl_blacs_intelmpi_lp64  -Wl,--end-group -lpthread -lm

all: static

static: dgeqrf2 dgemm3 bcast3 exafmm6 amg8 kripke9

dgeqrf2: dgeqrf2.cxx util.h ../src/model.h
	$(CXX) dgeqrf2.cxx -o dgeqrf2 $(CXXFLAGS)

dgemm3: dgemm3.cxx util.h ../src/model.h
	$(CXX) dgemm3.cxx -o dgemm3 $(CXXFLAGS)

bcast3: bcast3.cxx util.h ../src/model.h
	$(CXX) bcast3.cxx -o bcast3 $(CXXFLAGS)

exafmm6: exafmm6.cxx util.h ../src/model.h
	$(CXX) exafmm6.cxx -o exafmm6 $(CXXFLAGS)

amg8: amg8.cxx util.h ../src/model.h
	$(CXX) amg8.cxx -o amg8 $(CXXFLAGS)

kripke9: kripke9.cxx util.h ../src/model.h
	$(CXX) kripke9.cxx -o kripke9 $(CXXFLAGS)

clean:
	rm -f dgeqrf2 dgemm3 bcast3 exafmm6 amg8 kripke9
