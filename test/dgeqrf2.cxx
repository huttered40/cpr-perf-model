// Train data on a single process.
// Kernel/Application: dgeqrf
// Configuration parameters: 2 numerical (input) parameters (matrix dimensions m,n in LAPACK interface)
// Read training data from file provided at command line
// Set model hyperparameters via setting environment variables

#include <string>
#include <cstring>
#include <cassert>

#include "util.h"
#include "cp_perf_model.h"

void get_dataset(const char* dataset_file_path, int order, std::vector<double>& configurations, std::vector<double>& runtimes){
  std::ifstream my_file;
  my_file.open(dataset_file_path);

  std::string temp_num;
  // Read in column header
  getline(my_file,temp_num,'\n');

  // Input from columns {1,2}
  // Data from columns {3}
  while (getline(my_file,temp_num,',')){
    for (int i=0; i<order; i++){
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

  constexpr int nparam = 2;
  std::vector<performance_model::parameter_type> param_types(nparam,performance_model::parameter_type::NUMERICAL);
  char* dataset_file_path = argv[1];
  bool verbose = is_verbose();

  std::vector<double> configurations;
  std::vector<double> runtimes;
  get_dataset(argv[1],nparam,configurations,runtimes);
  assert(configurations.size() == runtimes.size()*nparam);

  std::vector<double> test_configurations;
  std::vector<double> test_runtimes;
  get_dataset(argv[2],nparam,test_configurations,test_runtimes);
  assert(test_runtimes.size()*nparam == test_configurations.size());

  performance_model::cpr_hyperparameter_pack interpolator_pack(nparam);
  set_cpr_param_pack(nparam,interpolator_pack,get_cpr_model_hyperparameter_options(),verbose);
  performance_model::cprg_hyperparameter_pack extrapolator_pack(nparam);
  set_cprg_param_pack(nparam,extrapolator_pack,verbose);

  performance_model::model* interpolator = new performance_model::cpr_model(nparam,&param_types[0],&interpolator_pack);
  performance_model::cprg_model* extrapolator = new performance_model::cprg_model(nparam,&param_types[0],&extrapolator_pack);

  performance_model::cpr_model_fit_info interpolator_fit_info;
  performance_model::cprg_model_fit_info extrapolator_fit_info;

  int nc = runtimes.size();
  const double* c = &configurations[0];
  const double* r = &runtimes[0];
  bool is_trained = interpolator->train(nc,c,r,false,&interpolator_fit_info);
  assert(is_trained);

  nc = runtimes.size();
  c = &configurations[0];
  r = &runtimes[0];
  is_trained = extrapolator->train(nc,c,r,false,&extrapolator_fit_info);
  assert(is_trained);

  interpolator->get_hyperparameters(interpolator_pack);
  extrapolator->get_hyperparameters(extrapolator_pack);

  print_model_info(interpolator_fit_info);
  print_model_info(extrapolator_fit_info);

  evaluate(nparam,test_runtimes.size(),test_runtimes,test_configurations,interpolator,extrapolator,verbose);

  if (argc>3){
    interpolator->write_to_file(argv[3]);
    interpolator->read_from_file(argv[3]);
  }
  if (argc>4){
    extrapolator->write_to_file(argv[4]);
    extrapolator->read_from_file(argv[4]);
  }

  delete interpolator;
  delete extrapolator;
  MPI_Finalize();
  return 0;
}
