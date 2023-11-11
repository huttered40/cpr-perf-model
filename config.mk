CXX=mpicxx
DEFS=
INCLUDES=-I$(WORK2)/../ctf_fork_fall_2023/ctf/include/
CXXFLAGS=-g -O3 -fopenmp $(DEFS) -std=c++0x -fPIC $(INCLUDES)

# To create a static library
#LDFLAGS=-L$(WORK2)/../ctf_fork_fall_2023/ctf/lib/ -Wl,-rpath,$(WORK2)/../ctf_fork_fall_2023/ctf/lib_shared/
LIBS=$(WORK2)/../ctf_fork_fall_2023/ctf/lib/libctf.a

# To create a shared library
#LDFLAGS=-L$(WORK2)/../ctf_fork_fall_2023/ctf/lib_shared/ -Wl,-rpath,$(WORK2)/../ctf_fork_fall_2023/ctf/lib_shared/
#LIBS=-lctf
