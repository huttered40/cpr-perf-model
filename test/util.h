#include <cstdio>
#include <fstream>
#include <string>

#include "cp_perf_model.h"

/*
#include <format>
#include <string_view>

template<typename... Args>
void print(const std::string_view fmt_str, Args&&... args){
  //auto fmt_args{std::make_format_args(args...)};// Interesting that we do not forward
  auto fmt_args{std::make_format_args(std::forward<Args>(args)...)};
  std::string outstr{ vformat(fmt_str,fmt_args) };
  fputs(outstr.c_str(),stdout);
}
*/

template<typename T>
void print(const char* msg, T val){
  std::cout << msg << val << "\n";
}

template<typename T, typename U>
void print(const char* msg, T val1, U val2){
  std::cout << msg << val1 << ": " << val2 << "\n";
}

void print_model_info(performance_model::cpr_model_fit_info& info){
  print("Number of distinct configurations: ",info.num_distinct_configurations);
  print("Number of tensor elements: ",info.num_tensor_elements);
  print("Tensor density: ",info.tensor_density);
  print("Loss: ",info.loss);
  print("Quadrature error: ",info.quadrature_error);
  print("Low-rank approximation error on observed tensor elements: ",info.low_rank_approximation_error);
  print("Training error: ",info.training_error);
}

void custom_assert(bool alert, const char* msg){
  if (!alert) fputs(msg,stdout);
}

double get_wall_time(){
  struct timeval time;
  if (gettimeofday(&time,NULL)){
    // Error
    return 0;
  }
  return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

std::vector<std::string> get_cpr_model_hyperparameter_options(){
  return {"CPPMI_PARTITION_SPACING",
          "CPPMI_PARTITIONS_PER_DIMENSION",
          "CPPMI_OBS_PER_PARTITION",
          "CPPMI_CP_RANK",
          "CPPMI_RUNTIME_TRANSFORM",
          "CPPMI_PARAMETER_TRANSFORM",
          "CPPMI_MAX_SPACING_FACTOR",
          "CPPMI_LOSS_FUNCTION",
          "CPPMI_REGULARIZATION",
          "CPPMI_MAX_NUM_RE_INITS",
          "CPPMI_OPTIMIZATION_CONVERGENCE_TOLERANCE_FOR_RE_INIT",
          "CPPMI_INTERPOLATION_FACTOR_TOL",
          "CPPMI_MIN_NUM_OBS_FOR_TRAINING",
          "CPPMI_OPTIMIZATION_BARRIER_START",
          "CPPMI_OPTIMIZATION_BARRIER_STOP",
          "CPPMI_OPTIMIZATION_BARRIER_REDUCTION_FACTOR",
          "CPPMI_FM_MAX_NUM_ITER",
          "CPPMI_FM_CONVERGENCE_TOL",
          "CPPMI_SWEEP_TOL",
          "CPPMI_MAX_NUM_SWEEPS",
          "CPPMI_AGGREGATE_OBS_ACROSS_COMM",
         };
}

void set_cpr_param_pack(int nparam, performance_model::cpr_hyperparameter_pack& arg_pack, std::vector<std::string>&& hyperparameter_options = get_cpr_model_hyperparameter_options(), bool verbose=false){

  assert(hyperparameter_options.size() == 21);
  // Set model hyper-parameters
  char* env_var_ptr = std::getenv(hyperparameter_options[0].c_str());
  if (env_var_ptr != NULL){
    //TODO: Replace with strcmp
    if (std::string(env_var_ptr) == "GEOMETRIC"){
      for (int i=0; i<nparam; i++) arg_pack._partition_spacing[i] = performance_model::parameter_range_partition::GEOMETRIC;
    }
    else if (std::string(env_var_ptr) == "UNIFORM"){
      for (int i=0; i<nparam; i++) arg_pack._partition_spacing[i] = performance_model::parameter_range_partition::UNIFORM;
    }
    else if (std::string(env_var_ptr) == "SINGLE"){
      for (int i=0; i<nparam; i++) arg_pack._partition_spacing[i] = performance_model::parameter_range_partition::SINGLE;
    }
    else if (std::string(env_var_ptr) == "AUTOMATIC"){
      for (int i=0; i<nparam; i++) arg_pack._partition_spacing[i] = performance_model::parameter_range_partition::AUTOMATIC;
    }
    else custom_assert(false,"Invalid option for ..\n");
  }

  env_var_ptr = std::getenv(hyperparameter_options[1].c_str());
  if (env_var_ptr != NULL){
    arg_pack._partitions_per_dimension = atoi(env_var_ptr);
    custom_assert(arg_pack._partitions_per_dimension>0, "Invalid option for CPPMI_PARTITIONS_PER_DIMENSION\n");
  }
  if (verbose) std::cout << arg_pack._partitions_per_dimension << "\n";

  env_var_ptr = std::getenv(hyperparameter_options[2].c_str());
  if (env_var_ptr != NULL){
    arg_pack._observations_per_partition = atoi(env_var_ptr);
    custom_assert(arg_pack._observations_per_partition>0, "Invalid option for CPPMI_OBS_PER_PARTITION\n");
  }
  if (verbose) std::cout << arg_pack._observations_per_partition << std::endl;

  for (int i=0; i<nparam; i++){
    if (arg_pack._partition_spacing[i] == performance_model::parameter_range_partition::AUTOMATIC){
      arg_pack._partition_info[i] = arg_pack._observations_per_partition;
    }
    else if (arg_pack._partition_spacing[i] == performance_model::parameter_range_partition::GEOMETRIC || arg_pack._partition_spacing[i] == performance_model::parameter_range_partition::UNIFORM){
      arg_pack._partition_info[i] = arg_pack._partitions_per_dimension;
    }
    else if (arg_pack._partition_spacing[i] == performance_model::parameter_range_partition::SINGLE){
      // if interval_spacing[i] == parameter_range_partition::SINGLE, its corresponding entry in partitions_per_dim_info is not parsed.
      arg_pack._partition_info[i] = -1;
    }
  }
  if (verbose){
    for (int i=0; i<nparam; i++) std::cout << arg_pack._partition_info[i] << " ";
    std::cout << "\n";
  }

  env_var_ptr = std::getenv(hyperparameter_options[3].c_str());
  if (env_var_ptr != NULL){
    arg_pack._cp_rank = atoi(env_var_ptr);
    custom_assert(arg_pack._cp_rank >= 1, "Invalid option for CPPMI_CP_RANK\n");
  }
  if (verbose) std::cout << arg_pack._cp_rank << "\n";

  env_var_ptr = std::getenv(hyperparameter_options[4].c_str());
  if (env_var_ptr != NULL){
    if (std::string(env_var_ptr) == "NONE") arg_pack._runtime_transformation = performance_model::runtime_transformation::NONE;
    else if (std::string(env_var_ptr) == "LOG") arg_pack._runtime_transformation = performance_model::runtime_transformation::LOG;
    else custom_assert(false,"Invalid option for CPPMI_RUNTIME_TRANSFORM\n");
    //custom_assert(arg_pack.response_transform>=0 && arg_pack.response_transform<=1, "Invalid option for CPPMI_RUNTIME_TRANSFORM\n");
  }
  //if (verbose) std::cout << arg_pack.response_transform << "\n";

  env_var_ptr = std::getenv(hyperparameter_options[5].c_str());
  if (env_var_ptr != NULL){
    if (std::string(env_var_ptr) == "NONE") arg_pack._parameter_transformation = performance_model::parameter_transformation::NONE;
    else if (std::string(env_var_ptr) == "LOG") arg_pack._parameter_transformation = performance_model::parameter_transformation::LOG;
    else custom_assert(false,"Invalid option for CPPMI_PARAMETER_TRANSFORM\n");
    //custom_assert(arg_pack.feature_transform>=0 && arg_pack.feature_transform<=1, "Invalid option for CPPMI_PARAMETER_TRANSFORM");
  }
  //if (verbose) std::cout << arg_pack.feature_transform << "\n";

  env_var_ptr = std::getenv(hyperparameter_options[6].c_str());
  if (env_var_ptr != NULL){
    arg_pack._max_partition_spacing_factor = atof(env_var_ptr);
    custom_assert(arg_pack._max_partition_spacing_factor > 1., "Invalid option for CPPMI_MAX_SPACING_FACTOR\n");
  }
  if (verbose) std::cout << arg_pack._max_partition_spacing_factor << "\n";

  env_var_ptr = std::getenv(hyperparameter_options[7].c_str());
  if (env_var_ptr != NULL){
    if (std::string(env_var_ptr) == "MSE") { arg_pack._loss_function = performance_model::loss_function::MSE; }
    else if (std::string(env_var_ptr) == "MLogQ2") { arg_pack._loss_function = performance_model::loss_function::MLOGQ2; }
    else custom_assert(false,"Invalid option for CPPMI_LOSS_FUNCTION\n");
  }
  //if (verbose) std::cout << arg_pack.loss_function_1 << "\n";

  env_var_ptr = std::getenv(hyperparameter_options[8].c_str());
  if (env_var_ptr != NULL){
    arg_pack._regularization = atof(env_var_ptr);
    custom_assert(arg_pack._regularization >= 0, "Invalid option for CPPMI_REGULARIZATION\n");
  }
  if (verbose) std::cout << arg_pack._regularization << "\n";

  env_var_ptr = std::getenv(hyperparameter_options[9].c_str());
  if (env_var_ptr != NULL){
    arg_pack._max_num_re_inits = atoi(env_var_ptr);
    custom_assert(arg_pack._max_num_re_inits >= 0, "Invalid option for CPPMI_MAX_NUM_RE_INITS\n");
  }
  if (verbose) std::cout << arg_pack._max_num_re_inits << "\n";

  env_var_ptr = std::getenv(hyperparameter_options[10].c_str());
  if (env_var_ptr != NULL){
    arg_pack._optimization_convergence_tolerance_for_re_init = atof(env_var_ptr);
    custom_assert(arg_pack._optimization_convergence_tolerance_for_re_init > 0, "Invalid option for CPPMI_OPTIMIZATION_CONVERGENCE_TOLERANCE_FOR_RE_INIT\n");
  }
  if (verbose) std::cout << arg_pack._optimization_convergence_tolerance_for_re_init << "\n";

  env_var_ptr = std::getenv(hyperparameter_options[11].c_str());
  if (env_var_ptr != NULL){
    arg_pack._interpolation_factor_tolerance = atof(env_var_ptr);
    custom_assert(arg_pack._interpolation_factor_tolerance >= 0, "Invalid option for CPPMI_INTERPOLATION_FACTOR_TOL\n");
  }
  if (verbose) std::cout << arg_pack._interpolation_factor_tolerance << "\n";

  env_var_ptr = std::getenv(hyperparameter_options[12].c_str());
  if (env_var_ptr != NULL){
    arg_pack._min_num_distinct_observed_configurations = atoi(env_var_ptr);
    custom_assert(arg_pack._min_num_distinct_observed_configurations >= 1, "Invalid option for CPPMI_MIN_NUM_OBS_FOR_TRAINING\n");
  }
  if (verbose) std::cout << arg_pack._min_num_distinct_observed_configurations << "\n";

  env_var_ptr = std::getenv(hyperparameter_options[13].c_str());
  if (env_var_ptr != NULL){
    arg_pack._optimization_barrier_start = atof(env_var_ptr);
    custom_assert(arg_pack._optimization_barrier_start >= 0, "Invalid option for CPPMI_OPTIMIZATION_BARRIER_START\n");
  }
  if (verbose) std::cout << arg_pack._optimization_barrier_start << "\n";

  env_var_ptr = std::getenv(hyperparameter_options[14].c_str());
  if (env_var_ptr != NULL){
    arg_pack._optimization_barrier_stop = atof(env_var_ptr);
    custom_assert(arg_pack._optimization_barrier_stop >= 0, "Invalid option for CPPMI_OPTIMIZATION_BARRIER_STOP\n");
    if (arg_pack._optimization_barrier_start > 0){
      custom_assert(arg_pack._optimization_barrier_stop < arg_pack._optimization_barrier_start, "Invalid option for CPPMI_OPTIMIZATION_BARRIER_STOP\n");
    }
  }
  if (verbose) std::cout << arg_pack._optimization_barrier_stop << "\n";

  env_var_ptr = std::getenv(hyperparameter_options[15].c_str());
  if (env_var_ptr != NULL){
    arg_pack._optimization_barrier_reduction_factor = atof(env_var_ptr);
    custom_assert(arg_pack._optimization_barrier_reduction_factor > 1, "Invalid option for CPPMI_BARRIER_REDUCTION_FACTOR\n");
  }
  if (verbose) std::cout << arg_pack._optimization_barrier_reduction_factor << "\n";

  env_var_ptr = std::getenv(hyperparameter_options[16].c_str());
  if (env_var_ptr != NULL){
    arg_pack._factor_matrix_optimization_max_num_iterations = atoi(env_var_ptr);
    custom_assert(arg_pack._factor_matrix_optimization_max_num_iterations >= 0, "Invalid option for CPPMI_FM_MAX_NUM_ITER\n");
  }
  if (verbose) std::cout << arg_pack._factor_matrix_optimization_max_num_iterations << "\n";

  env_var_ptr = std::getenv(hyperparameter_options[17].c_str());
  if (env_var_ptr != NULL){
    arg_pack._factor_matrix_optimization_convergence_tolerance = atof(env_var_ptr);
    custom_assert(arg_pack._factor_matrix_optimization_convergence_tolerance >=0, "Invalid option for CPPMI_FM_CONVERGENCE_TOL\n");
  }
  if (verbose) std::cout << arg_pack._factor_matrix_optimization_convergence_tolerance << "\n";

  env_var_ptr = std::getenv(hyperparameter_options[18].c_str());
  if (env_var_ptr != NULL){
    arg_pack._optimization_convergence_tolerance = atof(env_var_ptr);
    custom_assert(arg_pack._optimization_convergence_tolerance>0, "Invalid option for CPPMI_SWEEP_TOL\n");
  }
  if (verbose) std::cout << arg_pack._optimization_convergence_tolerance << "\n";

  env_var_ptr = std::getenv(hyperparameter_options[19].c_str());
  if (env_var_ptr != NULL){
    arg_pack._max_num_optimization_sweeps = atoi(env_var_ptr);
    custom_assert(arg_pack._max_num_optimization_sweeps>0, "Invalid option for CPPMI_MAX_NUM_SWEEPS\n");
  }
  if (verbose) std::cout << arg_pack._max_num_optimization_sweeps << "\n";

  env_var_ptr = std::getenv(hyperparameter_options[20].c_str());
  if (env_var_ptr != NULL){
    arg_pack._aggregate_obs_across_communicator = (1==atoi(env_var_ptr));
    //custom_assert(arg_pack._aggregate_obs_across_communicator>0, "Invalid option for CPPMI_AGGREGATE_OBS_ACROSS_COMM\n");
  }
  if (verbose) std::cout << arg_pack._aggregate_obs_across_communicator << "\n";
}

void set_cprg_param_pack(int nparam, performance_model::cprg_hyperparameter_pack& arg_pack, bool verbose=false){
  auto&& hyperparameter_options_ = get_cpr_model_hyperparameter_options();
  for (auto& it : hyperparameter_options_) it[4]='E';
  set_cpr_param_pack(nparam,arg_pack,std::move(hyperparameter_options_),verbose);

  char* env_var_ptr = std::getenv("CPPME_MAX_SPLINE_DEGREE");
  if (env_var_ptr != NULL){
    arg_pack._max_spline_degree = atoi(env_var_ptr);
    custom_assert(arg_pack._max_spline_degree>0, "Invalid option for CPPME_MAX_SPLINE_DEGREE\n");
  }
  if (verbose) std::cout << arg_pack._max_spline_degree << "\n";

  env_var_ptr = std::getenv("CPPME_FACTOR_MATRIX_ELEMENT_TRANSFORM");
  if (env_var_ptr != NULL){
    if (std::string(env_var_ptr) == "NONE") arg_pack._factor_matrix_element_transformation = performance_model::runtime_transformation::NONE;
    else if (std::string(env_var_ptr) == "LOG") arg_pack._factor_matrix_element_transformation = performance_model::runtime_transformation::LOG;
  }

  env_var_ptr = std::getenv("CPPME_FACTOR_MATRIX_UNDERLYING_POSITION_TRANSFORM");
  if (env_var_ptr != NULL){
    if (std::string(env_var_ptr) == "NONE") arg_pack._factor_matrix_underlying_position_transformation = performance_model::parameter_transformation::NONE;
    else if (std::string(env_var_ptr) == "LOG") arg_pack._factor_matrix_underlying_position_transformation = performance_model::parameter_transformation::LOG;
  }
}

bool is_verbose(){
  char* env_var_ptr = std::getenv("CPPM_VERBOSE");
  if (env_var_ptr != NULL){
    return 1==atoi(env_var_ptr);
  } else return false;
}

struct evaluation_info{
  double avg_inference_latency{0};
  double max_inference_latency{0};
  // mlogq: arithmetic mean of log-ratios
  double mlogq_error{0};
  double max_logq_error{0};
  // mlogqabs: arithmetic mean of absolute values of log-ratios
  double mlogqabs_error{0};
  double max_logqabs_error{0};
  // mlogq2: arithmetic mean of the square of log-ratios
  double mlogq2_error{0};
  double max_logq2_error{0};
  // maps: arithmetic mean of the absolute percentage error
  double maps_error{0};
  double max_aps_error{0};
};

void evaluate(int nparam, int size, std::vector<double>& runtimes, std::vector<double>& configurations, performance_model::model* interpolator, performance_model::model* extrapolator, bool verbose=false){
  evaluation_info info;
  for (int i=0; i<size; i++){
    double start_inference_latency = get_wall_time();
    double runtime_prediction = performance_model::predict(&configurations[i*nparam],interpolator,extrapolator);
    double inference_latency = get_wall_time() - start_inference_latency;
    info.avg_inference_latency += inference_latency;
    double local_logq_error = log(runtime_prediction / runtimes[i]);
    double local_logqabs_error = std::abs(log(runtime_prediction / runtimes[i]));
    double local_logq2_error = log(runtime_prediction / runtimes[i]); local_logq2_error *= local_logq2_error;
    double local_aps_error = std::abs(runtime_prediction-runtimes[i]) / runtimes[i];
    info.mlogq_error += local_logq_error;
    info.mlogqabs_error += local_logqabs_error;
    info.mlogq2_error += local_logq2_error;
    info.maps_error += local_aps_error;
    info.max_logq_error = std::max(local_logq_error,info.max_logq_error);
    info.max_logqabs_error = std::max(local_logqabs_error,info.max_logqabs_error);
    info.max_logq2_error = std::max(local_logq2_error,info.max_logq2_error);
    info.max_aps_error = std::max(local_aps_error,info.max_aps_error);
    info.max_inference_latency = std::max(inference_latency,info.max_inference_latency); 
    if (verbose){
      std::cout << runtimes[i] << " " << runtime_prediction << " " << local_logq_error << " " << local_logqabs_error << " " << local_logq2_error << " " << inference_latency << std::endl;
    }
  }
  info.mlogq_error /= size;
  info.mlogqabs_error /= size;
  info.mlogq2_error /= size;
  info.maps_error /= size;
  info.avg_inference_latency /= size;

  std::cout << "MLogQ prediction error: " << info.mlogq_error << "\n";
  std::cout << "Maximum MLogQ prediction error: " << info.max_logq_error << "\n";
  std::cout << "MLogQAbs prediction error: " << info.mlogqabs_error << "\n";
  std::cout << "Maximum MLogQAbs prediction error: " << info.max_logqabs_error << "\n";
  std::cout << "MLogQ2 prediction error: " << info.mlogq2_error << "\n";
  std::cout << "Maximum MLogQ2 prediction error: " << info.max_logq2_error << "\n";
  std::cout << "MAPS prediction error: " << info.maps_error << "\n";
  std::cout << "Maximum APS prediction error: " << info.max_aps_error << "\n";
  std::cout << "Average inference latency: " << info.avg_inference_latency << "\n"; 
  std::cout << "Maximum inference latency: " << info.max_inference_latency << "\n"; 
}
