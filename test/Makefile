CXX=mpicxx
DEFS=
INCLUDES=-I$(WORK2)/../cpr-perf-model/include/
CXXFLAGS=-g -O3 -fopenmp $(DEFS) -std=c++0x $(INCLUDES)

# Using static library
#LDFLAGS=-L../lib/ -L$(WORK2)/../ctf_fork_fall_2023/ctf/lib/ -Wl,-rpath,$(WORK2)/../ctf_fork_fall_2023/ctf/lib_shared/ -L/opt/intel/compilers_and_libraries_2018.2.199/linux/mkl/lib/intel64/
#LIBS=-lcpr_perf_model -lctf -lmkl_scalapack_lp64 -Wl,--start-group -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lmkl_blacs_intelmpi_lp64  -Wl,--end-group -lpthread -lm

# Using shared library
LDFLAGS=-L../lib/ -Wl,-rpath,../lib/# -L/opt/intel/compilers_and_libraries_2018.2.199/linux/mkl/lib/intel64/
LIBS=-lcpr_perf_model# -lmkl_scalapack_lp64 -Wl,--start-group -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lmkl_blacs_intelmpi_lp64  -Wl,--end-group -lpthread -lm

all: test_util.o dgeqrf2 dgemm3 bcast3 exafmm4 exafmm6 amg6 amg8 kripke7 kripke9 dgemm10 mm-blocked13 mm-blocked10

test_util.o: util.cxx util.h
	$(CXX) $(CXXFLAGS) util.cxx -c -o test_util.o

dgeqrf2: dgeqrf2.cxx test_util.o
	$(CXX) $(CXXFLAGS) dgeqrf2.cxx test_util.o $(LDFLAGS) $(LIBS) -o dgeqrf2

dgemm3: dgemm3.cxx test_util.o
	$(CXX) $(CXXFLAGS) dgemm3.cxx test_util.o $(LDFLAGS) $(LIBS) -o dgemm3

dgemm10: dgemm10.cxx test_util.o
	$(CXX) $(CXXFLAGS) dgemm10.cxx test_util.o $(LDFLAGS) $(LIBS) -o dgemm10

bcast3: bcast3.cxx test_util.o
	$(CXX) $(CXXFLAGS) bcast3.cxx test_util.o $(LDFLAGS) $(LIBS) -o bcast3

exafmm4: exafmm4.cxx test_util.o
	$(CXX) $(CXXFLAGS) exafmm4.cxx test_util.o $(LDFLAGS) $(LIBS) -o exafmm4

exafmm6: exafmm6.cxx test_util.o
	$(CXX) $(CXXFLAGS) exafmm6.cxx test_util.o $(LDFLAGS) $(LIBS) -o exafmm6

amg6: amg6.cxx test_util.o
	$(CXX) $(CXXFLAGS) amg6.cxx test_util.o $(LDFLAGS) $(LIBS) -o amg6

amg8: amg8.cxx test_util.o
	$(CXX) $(CXXFLAGS) amg8.cxx test_util.o $(LDFLAGS) $(LIBS) -o amg8

kripke7: kripke7.cxx test_util.o
	$(CXX) $(CXXFLAGS) kripke7.cxx test_util.o $(LDFLAGS) $(LIBS) -o kripke7

kripke9: kripke9.cxx test_util.o
	$(CXX) $(CXXFLAGS) kripke9.cxx test_util.o $(LDFLAGS) $(LIBS) -o kripke9

mm-blocked10: mm-blocked10.cxx test_util.o
	$(CXX) $(CXXFLAGS) mm-blocked10.cxx test_util.o $(LDFLAGS) $(LIBS) -o mm-blocked10

mm-blocked13: mm-blocked13.cxx test_util.o
	$(CXX) $(CXXFLAGS) mm-blocked13.cxx test_util.o $(LDFLAGS) $(LIBS) -o mm-blocked13

clean:
	rm -f test_util.o dgeqrf2 dgemm3 bcast3 exafmm6 amg8 kripke9 kripke7 amg6 exafmm4 dgemm10 mm-blocked13 mm-blocked10
