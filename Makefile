include config.mk

all: lib/libcpr_perf_model.a

static: lib/libcpr_perf_model.a

shared: lib/libcpr_perf_model.so

lib/libcpr_perf_model.a: obj/cp_model.o
	ar -x ../ctf_fork_fall_2023/ctf/lib/libctf.a
	ar -crs lib/libcpr_perf_model.a obj/cp_model.o *.o
	rm *.o

lib/libcpr_perf_model.so: obj/cp_model.o
	$(CXX) -shared -o lib/libcpr_perf_model.so obj/cp_model.o $(LDFLAGS) $(LIBS)

obj/cp_model.o: src/cp_model.cxx
	$(CXX) src/cp_model.cxx -c -o obj/cp_model.o $(CXXFLAGS)

clean:
	rm -f obj/*.o lib/libcpr_perf_model.a lib/libcpr_perf_model.so
