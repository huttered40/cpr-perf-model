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
	ar -crs lib/libcpr_perf_model.a obj/hyperparameter_pack.o obj/parameter_pack.o obj/cpr_hyperparameter_pack.o obj/cpr_parameter_pack.o obj/model.o obj/cpr_model.o obj/predict.o $(LIBS)

lib/libcpr_perf_model.so: obj/hyperparameter_pack.o obj/predict.o obj/model.o obj/cpr_hyperparameter_pack.o obj/parameter_pack.o obj/cpr_parameter_pack.o obj/cpr_model.o 
	$(CXX) -shared -o lib/libcpr_perf_model.so obj/model.o obj/cpr_hyperparameter_pack.o obj/predict.o obj/hyperparameter_pack.o obj/parameter_pack.o obj/cpr_parameter_pack.o obj/cpr_model.o $(LDFLAGS) $(LIBS)

obj/hyperparameter_pack.o: src/hyperparameter_pack.cxx
	$(CXX) $(CXXFLAGS) src/hyperparameter_pack.cxx -c -o obj/hyperparameter_pack.o

obj/cpr_hyperparameter_pack.o: src/cpr/cpr_hyperparameter_pack.cxx
	$(CXX) $(CXXFLAGS) src/cpr/cpr_hyperparameter_pack.cxx -c -o obj/cpr_hyperparameter_pack.o

obj/parameter_pack.o: src/parameter_pack.cxx
	$(CXX) $(CXXFLAGS) src/parameter_pack.cxx -c -o obj/parameter_pack.o

obj/cpr_parameter_pack.o: src/cpr/cpr_parameter_pack.cxx
	$(CXX) $(CXXFLAGS) src/cpr/cpr_parameter_pack.cxx -c -o obj/cpr_parameter_pack.o

obj/model.o: src/model.cxx
	$(CXX) $(CXXFLAGS) src/model.cxx -c -o obj/model.o

obj/cpr_model.o: src/cpr/cpr_model.cxx
	$(CXX) $(CXXFLAGS) src/cpr/cpr_model.cxx -c -o obj/cpr_model.o

obj/predict.o: src/predict.cxx
	$(CXX) $(CXXFLAGS) src/predict.cxx -c -o obj/predict.o

clean:
	rm -f obj/*.o lib/libcpr_perf_model.a lib/libcpr_perf_model.so
