include config.mk

all: lib/libcpr_perf_model.a

static: lib/libcpr_perf_model.a

shared: lib/libcpr_perf_model.so

lib/libcpr_perf_model.a:\
	obj/hyperparameter_pack.o\
	obj/predict.o\
	obj/model.o\
	obj/cpr_hyperparameter_pack.o\
	obj/parameter_pack.o\
	obj/cpr_parameter_pack.o\
	obj/cpr_model.o
	ar -x ../ctf_fork_fall_2023/ctf/lib/libctf.a
	ar -crs lib/libcpr_perf_model.a obj/predict.o obj/model.o obj/hyperparameter_pack.o obj/cpr_hyperparameter_pack.o obj/parameter_pack.o obj/cpr_parameter_pack.o obj/cpr_model.o *.o
	rm *.o

lib/libcpr_perf_model.so: obj/hyperparameter_pack.o obj/predict.o obj/model.o obj/cpr_hyperparameter_pack.o obj/parameter_pack.o obj/cpr_parameter_pack.o obj/cpr_mode.o 
	$(CXX) -shared -o lib/libcpr_perf_model.so obj.model.o obj/cpr_hyperparameter_pack.o obj/predict.o obj/hyperparameter_pack.o obj/parameter_pack.o obj/cpr_parameter_pack.o obj/cpr_model.o $(LDFLAGS) $(LIBS)

obj/hyperparameter_pack.o: src/hyperparameter_pack.cxx
	$(CXX) src/hyperparameter_pack.cxx -c -o obj/hyperparameter_pack.o $(CXXFLAGS)

obj/cpr_hyperparameter_pack.o: src/cpr/cpr_hyperparameter_pack.cxx
	$(CXX) src/cpr/cpr_hyperparameter_pack.cxx -c -o obj/cpr_hyperparameter_pack.o $(CXXFLAGS)

obj/parameter_pack.o: src/parameter_pack.cxx
	$(CXX) src/parameter_pack.cxx -c -o obj/parameter_pack.o $(CXXFLAGS)

obj/cpr_parameter_pack.o: src/cpr/cpr_parameter_pack.cxx
	$(CXX) src/cpr/cpr_parameter_pack.cxx -c -o obj/cpr_parameter_pack.o $(CXXFLAGS)

obj/model.o: src/model.cxx
	$(CXX) src/model.cxx -c -o obj/model.o $(CXXFLAGS)

obj/cpr_model.o: src/cpr/cpr_model.cxx
	$(CXX) src/cpr/cpr_model.cxx -c -o obj/cpr_model.o $(CXXFLAGS)

obj/predict.o: src/predict.cxx
	$(CXX) src/predict.cxx -c -o obj/predict.o $(CXXFLAGS)

clean:
	rm -f obj/*.o lib/libcpr_perf_model.a lib/libcpr_perf_model.so
