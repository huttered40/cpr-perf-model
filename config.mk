CXX=mpicxx
DEFS=
INCLUDES=-I$(WORK2)/../ctf_fork_fall_2023/ctf/include/
LDFLAGS=-L$(WORK2)/../ctf_fork_fall_2023/ctf/lib/
LIBS=-lctf -L/opt/intel/compilers_and_libraries_2018.2.199/linux/mkl/lib/intel64/ -lmkl_scalapack_lp64 -Wl,--start-group -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lmkl_blacs_intelmpi_lp64  -Wl,--end-group -lpthread -lm
CXXFLAGS=-g -O3 -fopenmp $(DEFS) -std=c++0x -fPIC $(INCLUDES)
