// Train data on a single process.
// Kernel/Application: dgemm
// Configuration parameters: 10 tuning parameters
// Read training data from file provided at command line
// Set model hyperparameters via setting environment variables

#include <string>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <iostream>

#include "util.h"
#include "cp_perf_model.h"

void get_dataset(const char* dataset_file_path, size_t order, std::vector<double>& configurations, std::vector<double>& runtimes, size_t start_idx, size_t end_idx){
  std::ifstream my_file;
  my_file.open(dataset_file_path);

  std::string temp_num;
/*
  // Read in column header
  getline(my_file,temp_num,'\n');
*/
  // Input from columns {1,2,3}
  // Data from columns {4}
  size_t idx = 0;
  while (getline(my_file,temp_num,',')){
    if (!(idx >= start_idx && idx < end_idx)) { getline(my_file,temp_num,'\n'); idx++; continue; }
    for (size_t i=0; i<3; i++) getline(my_file,temp_num,',');
    for (size_t i=0; i<order; i++){
      getline(my_file,temp_num,',');
      configurations.push_back(atof(temp_num.c_str()));
    }
    getline(my_file,temp_num,',');
    getline(my_file,temp_num,',');
    runtimes.push_back(atof(temp_num.c_str()));
    getline(my_file,temp_num,'\n');// read in rest of line
    idx++;
  }
}


int main(int argc, char** argv){
  // Read in datafile stored in CSV (comma-delimited) format
  // Just train and evaluate training error

  MPI_Init(&argc,&argv);

  constexpr size_t nparam = 10;
  std::vector<performance_model::parameter_type> param_types(nparam,performance_model::parameter_type::CATEGORICAL);
  bool verbose = is_verbose();

  performance_model::cpr_hyperparameter_pack interpolator_pack(nparam);
  set_cpr_param_pack(nparam,interpolator_pack,get_cpr_model_hyperparameter_options(),verbose);
  performance_model::cprg_hyperparameter_pack extrapolator_pack(nparam);
  set_cprg_param_pack(nparam,extrapolator_pack,verbose);

  for (size_t i=0; i<nparam; i++){
    interpolator_pack.partition_spacing[i] = performance_model::parameter_range_partition::SINGLE;
    extrapolator_pack.partition_spacing[i] = performance_model::parameter_range_partition::SINGLE;
  }

  std::vector<double> configurations;
  std::vector<double> runtimes;
  get_dataset(argv[1],nparam,configurations,runtimes,atoi(argv[3]),atoi(argv[4]));
  assert(configurations.size() == runtimes.size()*nparam);

  std::vector<double> test_configurations;
  std::vector<double> test_runtimes;
  get_dataset(argv[2],nparam,test_configurations,test_runtimes,atoi(argv[5]),atoi(argv[6]));
  assert(test_runtimes.size()*nparam == test_configurations.size());

  performance_model::model* interpolator = new performance_model::cpr_model(nparam,&param_types[0],&interpolator_pack);
  performance_model::cprg_model* extrapolator = new performance_model::cprg_model(nparam,&param_types[0],&extrapolator_pack);

  performance_model::cpr_model_fit_info interpolator_fit_info;
  performance_model::cprg_model_fit_info extrapolator_fit_info;

  size_t nc = runtimes.size();
  if (argc>8){
    if (std::stoul(argv[8])<nc){
      nc = std::stoul(argv[8]);
      shuffle_runtimes(nc,nparam,runtimes,configurations);
    }
  }
  size_t nc2=nc;
  const double* c = &configurations[0];
  const double* r = &runtimes[0];
  bool is_trained = interpolator->train(nc,c,r,&interpolator_fit_info);
  assert(is_trained);

  c = &configurations[0];
  r = &runtimes[0];
  is_trained = extrapolator->train(nc2,c,r,&extrapolator_fit_info);
  assert(is_trained);

  interpolator->get_hyperparameters(interpolator_pack);
  extrapolator->get_hyperparameters(extrapolator_pack);

  print_model_info(interpolator_fit_info);
  print_model_info(extrapolator_fit_info);

  evaluate(nparam,test_runtimes.size(),test_runtimes,test_configurations,interpolator,extrapolator,
           interpolator_pack,extrapolator_pack,interpolator_fit_info,extrapolator_fit_info,argv[7],verbose);

  if (argc>9){
    interpolator->write_to_file(argv[9]);
    interpolator->read_from_file(argv[9]);
  }
  if (argc>10){
    extrapolator->write_to_file(argv[10]);
    extrapolator->read_from_file(argv[10]);
  }

  delete interpolator;
  delete extrapolator;
  MPI_Finalize();
  return 0;
}
