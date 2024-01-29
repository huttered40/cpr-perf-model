// Train data on a single process.
// Kernel/Application: matrix multiplication with multi-level blocking
// Configuration parameters: 13 tuning parameters
// Read training data from file provided at command line
// Set model hyperparameters via setting environment variables

#include <string>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <iostream>

#include "util.h"
#include "cp_perf_model.h"

void get_dataset(const char* dataset_file_path, int order, std::vector<double>& configurations, std::vector<double>& runtimes){
  std::ifstream my_file;
  my_file.open(dataset_file_path);

  std::string temp_num;
  // Read in column header
  getline(my_file,temp_num,'\n');
  // Input from columns {1,2,3,4,5,6,10,11,12,13}
  // Data from columns {15}
  int idx = 0;
  double min_matrix_dimension = 10000000;
  double max_matrix_dimension = 0;
  while (getline(my_file,temp_num,',')){
    for (int i=0; i<6; i++){
      getline(my_file,temp_num,',');
      configurations.push_back(atof(temp_num.c_str()));
    }
    for (int i=0; i<3; i++){
      getline(my_file,temp_num,',');
    }
    for (int i=6; i<order; i++){
      getline(my_file,temp_num,',');
      configurations.push_back(atof(temp_num.c_str()));
    }
    getline(my_file,temp_num,',');// thread count
    getline(my_file,temp_num,',');
    runtimes.push_back(atof(temp_num.c_str()));
    getline(my_file,temp_num,'\n');// read in rest of line
    idx++;
/*
    for (int i=0; i<order; i++) std::cout << configurations[configurations.size()-order+i] << " ";
    std::cout << runtimes[runtimes.size()-1] << "\n";
*/
    for (int i=0; i<3; i++){
      if (min_matrix_dimension > configurations[configurations.size()-order+i]) min_matrix_dimension = configurations[configurations.size()-order+i];
      if (max_matrix_dimension < configurations[configurations.size()-order+i]) max_matrix_dimension = configurations[configurations.size()-order+i];
    }
  }
  std::cout << "Total number of runtimes - " << runtimes.size() << " " << configurations.size() << std::endl;
  std::cout << "Min runtime: " << *std::min_element(runtimes.begin(),runtimes.end()) << "\n";
  std::cout << "Max runtime: " << *std::max_element(runtimes.begin(),runtimes.end()) << "\n";
  std::cout << "Min matrix dimension: " << min_matrix_dimension << std::endl;
  std::cout << "Max matrix dimension: " << max_matrix_dimension << std::endl;
}


int main(int argc, char** argv){
  // Read in datafile stored in CSV (comma-delimited) format
  // Just train and evaluate training error

  MPI_Init(&argc,&argv);

  constexpr int nparam = 10;
  std::vector<performance_model::parameter_type> param_types(nparam,performance_model::parameter_type::CATEGORICAL);
  for (int i=0; i<6; i++){
    param_types[i] = performance_model::parameter_type::NUMERICAL;
  }
  bool verbose = is_verbose();

  performance_model::cpr_hyperparameter_pack interpolator_pack(nparam);
  set_cpr_param_pack(nparam,interpolator_pack,get_cpr_model_hyperparameter_options(),verbose);
  performance_model::cprg_hyperparameter_pack extrapolator_pack(nparam);
  set_cprg_param_pack(nparam,extrapolator_pack,verbose);

  for (int i=0; i<3; i++){
    interpolator_pack.partition_spacing[i] = performance_model::parameter_range_partition::GEOMETRIC;
    extrapolator_pack.partition_spacing[i] = performance_model::parameter_range_partition::GEOMETRIC;
  }
  for (int i=3; i<6; i++){
    interpolator_pack.partition_spacing[i] = performance_model::parameter_range_partition::UNIFORM;
    extrapolator_pack.partition_spacing[i] = performance_model::parameter_range_partition::UNIFORM;
    interpolator_pack.partition_info[i]=2;
    extrapolator_pack.partition_info[i]=2;
  }
  for (int i=6; i<nparam; i++){
    interpolator_pack.partition_spacing[i] = performance_model::parameter_range_partition::SINGLE;
    extrapolator_pack.partition_spacing[i] = performance_model::parameter_range_partition::SINGLE;
  }

  std::vector<double> configurations;
  std::vector<double> runtimes;
  get_dataset(argv[1],nparam,configurations,runtimes);
  assert(configurations.size() == runtimes.size()*nparam);

  std::vector<double> test_configurations;
  std::vector<double> test_runtimes;
  get_dataset(argv[2],nparam,test_configurations,test_runtimes);
  assert(test_runtimes.size()*nparam == test_configurations.size());

  performance_model::model* interpolator = new performance_model::cpr_model(nparam,&param_types[0],&interpolator_pack);
  performance_model::cprg_model* extrapolator = new performance_model::cprg_model(nparam,&param_types[0],&extrapolator_pack);

  performance_model::cpr_model_fit_info interpolator_fit_info;
  performance_model::cprg_model_fit_info extrapolator_fit_info;

  int nc = runtimes.size();
  if (argc>4){
    if (atoi(argv[4])<nc){
      nc = atoi(argv[4]);
      shuffle_runtimes(nc,nparam,runtimes,configurations);
    }
  }
  int nc2=nc;
  const double* c = &configurations[0];
  const double* r = &runtimes[0];
  bool is_trained = interpolator->train(nc,c,r,false,&interpolator_fit_info);
  assert(is_trained);

  c = &configurations[0];
  r = &runtimes[0];
  is_trained = extrapolator->train(nc2,c,r,false,&extrapolator_fit_info);
  assert(is_trained);

  interpolator->get_hyperparameters(interpolator_pack);
  extrapolator->get_hyperparameters(extrapolator_pack);

  print_model_info(interpolator_fit_info);
  print_model_info(extrapolator_fit_info);

  evaluate(nparam,test_runtimes.size(),test_runtimes,test_configurations,interpolator,extrapolator,
           interpolator_pack,extrapolator_pack,interpolator_fit_info,extrapolator_fit_info,argv[3],verbose);

  if (argc>5){
    interpolator->write_to_file(argv[5]);
    interpolator->read_from_file(argv[5]);
  }
  if (argc>6){
    extrapolator->write_to_file(argv[6]);
    extrapolator->read_from_file(argv[6]);
  }

  delete interpolator;
  delete extrapolator;
  MPI_Finalize();
  return 0;
}
