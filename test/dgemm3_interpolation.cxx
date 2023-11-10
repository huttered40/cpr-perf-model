// Train data on a single process.
// Kernel/Application: dgemm
// Configuration parameters: 3 numerical (input) parameters (matrix dimensions m,n,k in BLAS interface)
// Read training data from file provided at command line
// Set model hyperparameters via setting environment variables

#include <time.h>
#include <sys/time.h>
#include <string>
#include <cstring>
#include <cassert>
#include <cmath>
#include <iostream>

#include "util.h"

void get_dataset(const char* dataset_file_path, int order, std::vector<double>& configurations, std::vector<double>& runtimes){
  std::ifstream my_file;
  my_file.open(dataset_file_path);

  std::string temp_num;
  // Read in column header
  for (int i=0; i<order+1; i++){
    getline(my_file,temp_num,',');
  }
  getline(my_file,temp_num,'\n');

  while (getline(my_file,temp_num,',')){
    getline(my_file,temp_num,',');
    configurations.push_back(atof(temp_num.c_str()));
    for (int i=1; i<order; i++){
      getline(my_file,temp_num,',');
      configurations.push_back(atof(temp_num.c_str()));
    }
    getline(my_file,temp_num,',');
    runtimes.push_back(atof(temp_num.c_str()));
    getline(my_file,temp_num,'\n');// read in standard deviation
  }
}


int main(int argc, char** argv){
  // Read in datafile stored in CSV (comma-delimited) format
  // Just train and evaluate training error

  MPI_Init(&argc,&argv);

  constexpr int nparam = 3;
  std::vector<performance_model::parameter_type> param_types(nparam,performance_model::parameter_type::NUMERICAL);
  char* env_var_ptr;
  char* dataset_file_path = argv[1];
  std::vector<double> configurations;
  std::vector<double> runtimes;
  get_dataset(argv[1],nparam,configurations,runtimes);
  assert(configurations.size() == runtimes.size()*nparam);

  bool verbose = is_verbose();
  performance_model::cpr_hyperparameter_pack interpolator_pack(nparam);
  set_cpr_param_pack(nparam,interpolator_pack,get_cpr_model_hyperparameter_options(),verbose);
  performance_model::cprg_hyperparameter_pack extrapolator_pack(nparam);
  set_cprg_param_pack(nparam,extrapolator_pack,verbose);

  performance_model::model* interpolator = new performance_model::cpr_model(nparam,&param_types[0],&interpolator_pack);
  performance_model::cprg_model* extrapolator = new performance_model::cprg_model(nparam,&param_types[0],&extrapolator_pack);
  int nc = runtimes.size();
  const double* c = &configurations[0];
  const double* r = &runtimes[0];
  bool is_trained = interpolator->train(nc,c,r);
  assert(is_trained);
  is_trained = extrapolator->train(nc,c,r);
  assert(is_trained);
  interpolator->get_hyperparameters(interpolator_pack);
  extrapolator->get_hyperparameters(extrapolator_pack);
  if (verbose) print_model_info(nparam,interpolator_pack._info);
  if (verbose) print_model_info(nparam,extrapolator_pack._info);

  std::vector<double> test_configurations;
  std::vector<double> test_runtimes;
  get_dataset(argv[2],nparam,test_configurations,test_runtimes);
  assert(test_runtimes.size()*nparam == test_configurations.size());

  double total_err = 0;
  double max_err = 0;
  for (int i=0; i<test_runtimes.size(); i++){
    double runtime_prediction = performance_model::predict(&test_configurations[i*nparam],interpolator,extrapolator);
    double local_err = std::abs(log(runtime_prediction / test_runtimes[i]));
    total_err += local_err; 
    max_err = std::max(local_err,max_err);
    std::cout << test_runtimes[i] << " " << runtime_prediction << " " << local_err << " " << max_err << " " << log(runtime_prediction / test_runtimes[i]) << " " << std::abs(log(runtime_prediction / test_runtimes[i])) << std::endl;
  }
  total_err /= test_runtimes.size();
  std::cout << total_err << " " << max_err << std::endl;

  delete interpolator;
  delete extrapolator;
  MPI_Finalize();
  return 0;
}
