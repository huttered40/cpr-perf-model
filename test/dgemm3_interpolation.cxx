// Train data on a single MPI process.
// Kernel/Application: dgemm
// Configuration parameters: 3 numerical (input) parameters (matrix dimensions m,n,k in BLAS interface)
// Read training data from file provided at command line
// Set model hyperparameters via setting environment variables

#include <time.h>
#include <sys/time.h>
#include <string>
#include <cstring>
#include <iostream>

#include "cpr_perf_model.h"
#include "util.h"

int main(int argc, char** argv){
  // Read in datafile stored in CSV (comma-delimited) format
  // Just train and evaluate training error
  MPI_Init(&argc,&argv);

  constexpr int nparam = 3;
  std::vector<MODEL_PARAM_TYPE> param_types(nparam,NUMERICAL);
  char* env_var_ptr;
  double model_info[8+nparam];
  char* dataset_file_path = argv[1];
  std::vector<double> configurations;
  std::vector<double> runtimes;
  get_dataset(argv[1],nparam,configurations,runtimes);
  int nconfigurations = runtimes.size();// TODO
  assert(configurations.size() == runtimes.size()*nparam);

  bool verbose = false;
  env_var_ptr = std::getenv("CPPM_VERBOSE");
  if (env_var_ptr != NULL){
    verbose = 1==atoi(env_var_ptr);
  }

  // Set model hyper-parameters
  std::vector<MODEL_PARAM_SPACING> interval_spacing(nparam,GEOMETRIC);
  env_var_ptr = std::getenv("CPPM_INTERVAL_SPACING");
  if (env_var_ptr != NULL){
    if (std::string(env_var_ptr) == "GEOMETRIC"){ }
    else if (std::string(env_var_ptr) == "UNIFORM"){ interval_spacing.resize(nparam,UNIFORM); }
    else if (std::string(env_var_ptr) == "SINGLE"){ interval_spacing.resize(nparam,SINGLE); }
    else if (std::string(env_var_ptr) == "AUTOMATIC"){ interval_spacing.resize(nparam,AUTOMATIC); }
    else custom_assert(false,"Invalid option for CPPM_INTERVAL_SPACING");
  }
  if (verbose){
    for (int i=0; i<nparam; i++) std::cout << interval_spacing[i] << " ";
    std::cout << "\n";
  }

  int cells_per_dimension = 16;
  env_var_ptr = std::getenv("CPPM_CELLS_PER_DIMENSION");
  if (env_var_ptr != NULL){
    cells_per_dimension = atoi(env_var_ptr);
    custom_assert(cells_per_dimension>0, "Invalid option for CPPM_CELLS_PER_DIMENSION");
  }
  if (verbose) std::cout << cells_per_dimension << "\n";

  int observations_per_cell = 32;
  env_var_ptr = std::getenv("CPPM_OBS_PER_CELL");
  if (env_var_ptr != NULL){
    observations_per_cell = atoi(env_var_ptr);
    custom_assert(observations_per_cell>0, "Invalid option for CPPM_OBS_PER_CELL");
  }
  if (verbose) std::cout << observations_per_cell << std::endl;

  std::vector<int> cells_per_dim_info(nparam,-1);
  for (int i=0; i<interval_spacing.size(); i++){
    if (interval_spacing[i] == AUTOMATIC){
      cells_per_dim_info[i] = (-1)*observations_per_cell;
    }
    else if (interval_spacing[i] == GEOMETRIC || interval_spacing[i] == UNIFORM){
      cells_per_dim_info[i] = cells_per_dimension;
    }
    // if interval_spacing[i] == SINGLE, its corresponding entry in cells_per_dim_info is not parsed.
  }
  if (verbose){
    for (int i=0; i<nparam; i++) std::cout << cells_per_dim_info[i] << " ";
    std::cout << "\n";
  }

  int cp_rank_1 = 3;
  env_var_ptr = std::getenv("CPPM_CP_RANK_1");
  if (env_var_ptr != NULL){
    cp_rank_1 = atoi(env_var_ptr);
    custom_assert(cp_rank_1 >= 1, "Invalid option for CPPM_CP_RANK_1");
  }
  if (verbose) std::cout << cp_rank_1 << "\n";

  int cp_rank_2 = 1;
  env_var_ptr = std::getenv("CPPM_CP_RANK_2");
  if (env_var_ptr != NULL){
    cp_rank_2 = atoi(env_var_ptr);
    custom_assert(cp_rank_2 >= 1, "Invalid option for CPPM_CP_RANK_2");
  }
  if (verbose) std::cout << cp_rank_2 << "\n";

  int response_transform = 1;
  env_var_ptr = std::getenv("CPPM_RESPONSE_TRANSFORM_ID");
  if (env_var_ptr != NULL){
    response_transform = atoi(env_var_ptr);
    custom_assert(response_transform>=0 && response_transform<=1, "Invalid option for CPPM_RESPONSE_TRANSFORM_ID");
  }
  if (verbose) std::cout << response_transform << "\n";

  double max_spacing_factor = 2;// Only relevant for AUTOMATIC spacing
  env_var_ptr = std::getenv("CPPM_MAX_SPACING_FACTOR");
  if (env_var_ptr != NULL){
    max_spacing_factor = atof(env_var_ptr);
    custom_assert(max_spacing_factor > 1., "Invalid option for CPPM_MAX_SPACING_FACTOR");
  }
  if (verbose) std::cout << max_spacing_factor << "\n";

  char* loss_function = "MSE";
  env_var_ptr = std::getenv("CPPM_LOSS_FUNCTION");
  if (env_var_ptr != NULL){
    if (std::string(env_var_ptr) == "MSE") {}
    else if (std::string(env_var_ptr) == "MLogQ2") { strcpy(loss_function,env_var_ptr); }
    else custom_assert(false,"Invalid option for CPPM_LOSS_FUNCTION");
  }
  if (verbose) std::cout << loss_function << "\n";

  double reg_1 = 1e-3;
  env_var_ptr = std::getenv("CPPM_REG_1");
  if (env_var_ptr != NULL){
    reg_1 = atof(env_var_ptr);
    custom_assert(reg_1 >= 0, "Invalid option for CPPM_REG_1");
  }
  if (verbose) std::cout << reg_1 << "\n";

  double reg_2 = 1e-3;
  env_var_ptr = std::getenv("CPPM_REG_2");
  if (env_var_ptr != NULL){
    reg_2 = atof(env_var_ptr);
    custom_assert(reg_2 >= 0, "Invalid option for CPPM_REG_2");
  }
  if (verbose) std::cout << reg_2 << "\n";

  double barrier_start = 1e1;
  env_var_ptr = std::getenv("CPPM_BARRIER_START");
  if (env_var_ptr != NULL){
    barrier_start = atof(env_var_ptr);
    custom_assert(barrier_start >= 0, "Invalid option for CPPM_BARRIER_START");
  }
  if (verbose) std::cout << barrier_start << "\n";

  double barrier_stop = 1e-11;
  env_var_ptr = std::getenv("CPPM_BARRIER_STOP");
  if (env_var_ptr != NULL){
    barrier_stop = atof(env_var_ptr);
    custom_assert(barrier_stop >= 0, "Invalid option for CPPM_BARRIER_STOP");
    if (barrier_start > 0){
      custom_assert(barrier_stop < barrier_start, "Invalid option for CPPM_BARRIER_STOP");
    }
  }
  if (verbose) std::cout << barrier_stop << "\n";

  double barrier_reduction_factor = 8;
  env_var_ptr = std::getenv("CPPM_BARRIER_REDUCTION_FACTOR");
  if (env_var_ptr != NULL){
    barrier_reduction_factor = atof(env_var_ptr);
    custom_assert(barrier_reduction_factor > 1, "Invalid option for CPPM_BARRIER_REDUCTION_FACTOR");
  }
  if (verbose) std::cout << barrier_reduction_factor << "\n";

  int fm_max_num_iter = 10;
  env_var_ptr = std::getenv("CPPM_FM_MAX_NUM_ITER");
  if (env_var_ptr != NULL){
    fm_max_num_iter = atoi(env_var_ptr);
    custom_assert(fm_max_num_iter >= 0, "Invalid option for CPPM_FM_MAX_NUM_ITER");
  }
  if (verbose) std::cout << fm_max_num_iter << "\n";

  double fm_convergence_tol = 1e-3;
  env_var_ptr = std::getenv("CPPM_FM_CONVERGENCE_TOL");
  if (env_var_ptr != NULL){
    fm_convergence_tol = atof(env_var_ptr);
    custom_assert(fm_convergence_tol >=0, "Invalid option for CPPM_FM_CONVERGENCE_TOL");
  }
  if (verbose) std::cout << fm_convergence_tol << "\n";

  double sweep_tol_1 = 1e-2;
  env_var_ptr = std::getenv("CPPM_SWEEP_TOL_1");
  if (env_var_ptr != NULL){
    sweep_tol_1 = atof(env_var_ptr);
    custom_assert(sweep_tol_1>0, "Invalid option for CPPM_SWEEP_TOL_1");
  }
  if (verbose) std::cout << sweep_tol_1 << "\n";

  double sweep_tol_2 = 1e-2;
  env_var_ptr = std::getenv("CPPM_SWEEP_TOL_2");
  if (env_var_ptr != NULL){
    sweep_tol_2 = atof(env_var_ptr);
    custom_assert(sweep_tol_2>0, "Invalid option for CPPM_SWEEP_TOL_2");
  }
  if (verbose) std::cout << sweep_tol_2 << "\n";

  int max_num_sweeps_1 = 10;
  env_var_ptr = std::getenv("CPPM_MAX_NUM_SWEEPS_1");
  if (env_var_ptr != NULL){
    max_num_sweeps_1 = atoi(env_var_ptr);
    custom_assert(max_num_sweeps_1>0, "Invalid option for CPPM_MAX_NUM_SWEEPS_1");
  }
  if (verbose) std::cout << max_num_sweeps_1 << "\n";

  int max_num_sweeps_2 = 10;
  env_var_ptr = std::getenv("CPPM_MAX_NUM_SWEEPS_2");
  if (env_var_ptr != NULL){
    max_num_sweeps_2 = atoi(env_var_ptr);
    custom_assert(max_num_sweeps_2>0, "Invalid option for CPPM_MAX_NUM_SWEEPS_2");
  }
  if (verbose) std::cout << max_num_sweeps_2 << "\n";

  int max_spline_degree = 1;
  env_var_ptr = std::getenv("CPPM_MAX_SPLINE_DEGREE");
  if (env_var_ptr != NULL){
    max_spline_degree = atoi(env_var_ptr);
    custom_assert(max_spline_degree>0, "Invalid option for CPPM_MAX_SPLINE_DEGREE");
  }
  if (verbose) std::cout << max_spline_degree << "\n";

  bool aggregate_obs_across_comm = false;// This test uses a single MPI process

  cp_perf_model cpr(nparam,param_types,interval_spacing,{cp_rank_1,cp_rank_2},response_transform,false);

  bool is_trained = cpr.train(verbose,&model_info[0],nconfigurations,&configurations[0],&runtimes[0],MPI_COMM_SELF,MPI_COMM_WORLD,cells_per_dim_info,loss_function,aggregate_obs_across_comm,max_spacing_factor,{reg_1,reg_2},
                                  max_spline_degree,{sweep_tol_1,sweep_tol_2},{max_num_sweeps_1,max_num_sweeps_2},
                                  fm_convergence_tol,fm_max_num_iter,{barrier_stop,barrier_start},barrier_reduction_factor);

  if (verbose) print_model_info(nparam,model_info);
  if (verbose){
    std::cout << cpr.timer1 << " " << cpr.timer2 << " " << cpr.timer3 << " " << cpr.timer4 << " " << cpr.timer5 << " " << cpr.timer6 << " " << cpr.timer7 << std::endl;
  }

  MPI_Finalize();
  return 0;
}
