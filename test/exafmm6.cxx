// Train data on a single process.
// Kernel/Application: exafmm
// Configuration parameters: 6 numerical and categorical (input) parameters
// Read training data from file provided at command line
// Set model hyperparameters via setting environment variables

#include <string>
#include <cstring>
#include <cassert>

#include "util.h"
#include "cp_perf_model.h"

void get_dataset(const char* dataset_file_path, size_t order, std::vector<double>& configurations, std::vector<double>& runtimes){
  std::ifstream my_file;
  my_file.open(dataset_file_path);

  std::string temp_num;
  // Read in column header
  getline(my_file,temp_num,'\n');

  // Input from columns {2,3,4,6,7,8}
  // Data from columns {15}
  while (getline(my_file,temp_num,',')){
    getline(my_file,temp_num,',');
    for (size_t i=0; i<order+1; i++){
      getline(my_file,temp_num,',');
      if (i!=3) configurations.push_back(atof(temp_num.c_str()));
    }
    for (size_t i=0; i<6; i++) getline(my_file,temp_num,',');
    getline(my_file,temp_num,',');
    runtimes.push_back(atof(temp_num.c_str()));
    getline(my_file,temp_num,'\n');// read in rest
  }
}


int main(int argc, char** argv){
  // Read in datafile stored in CSV (comma-delimited) format
  // Just train and evaluate training error

  MPI_Init(&argc,&argv);

  constexpr size_t nparam = 6;
  std::vector<performance_model::parameter_type> param_types(nparam,performance_model::parameter_type::NUMERICAL);
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
  for (size_t i=0; i<nparam; i++){
    if (i==2){
      interpolator_pack.partition_spacing[i] = performance_model::parameter_range_partition::GEOMETRIC;
      extrapolator_pack.partition_spacing[i] = performance_model::parameter_range_partition::GEOMETRIC;
      continue;
    }
    else if (i==3){
      interpolator_pack.partition_spacing[i] = performance_model::parameter_range_partition::UNIFORM;
      extrapolator_pack.partition_spacing[i] = performance_model::parameter_range_partition::UNIFORM;
      continue;
    }
    interpolator_pack.partition_spacing[i] = performance_model::parameter_range_partition::SINGLE;
    extrapolator_pack.partition_spacing[i] = performance_model::parameter_range_partition::SINGLE;
  }

  performance_model::model* interpolator = new performance_model::cpr_model(nparam,&param_types[0],&interpolator_pack);
  performance_model::cprg_model* extrapolator = new performance_model::cprg_model(nparam,&param_types[0],&extrapolator_pack);

  performance_model::cpr_model_fit_info interpolator_fit_info;
  performance_model::cprg_model_fit_info extrapolator_fit_info;

  size_t nc = runtimes.size();
  if (argc>4){
    if (std::stoul(argv[4])<nc){
      nc = std::stoul(argv[4]);
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
