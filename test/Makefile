CXX=mpicxx
DEFS=
INCLUDES=-I$(WORK2)/../cpr-perf-model/include/
LDFLAGS=-L$(WORK2)/../cpr-perf-model/lib/
LIBS=-lcpr_perf_model
CXXFLAGS=-g -O3 -fopenmp $(DEFS) -std=c++0x $(INCLUDES) $(LDFLAGS) $(LIBS) -L/opt/intel/compilers_and_libraries_2018.2.199/linux/mkl/lib/intel64/ -lmkl_scalapack_lp64 -Wl,--start-group -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lmkl_blacs_intelmpi_lp64  -Wl,--end-group -lpthread -lm

all: static

static: dgemm3_interpolation

dgemm3_interpolation: dgemm3_interpolation.cxx util.h ../src/model.h
	$(CXX) dgemm3_interpolation.cxx -o dgemm3_interpolation $(CXXFLAGS)

clean:
	rm -f dgemm3_interpolation
