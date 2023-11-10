#include <cassert>

#include "cpr_hyperparameter_pack.h"
#include "cpr_types.h"
#include "../types.h"

namespace performance_model{

piecewise_hyperparameter_pack::piecewise_hyperparameter_pack(int nparam) : hyperparameter_pack(nparam){
  // Default
  this->_partitions_per_dimension=8;
  this->_observations_per_partition=32;
  this->_runtime_transformation = runtime_transformation::NONE;
  this->_parameter_transformation = parameter_transformation::NONE;
  this->_max_partition_spacing_factor=2;
  this->_cm_training=MPI_COMM_SELF;
  this->_cm_data=MPI_COMM_SELF;
  this->_aggregate_obs_across_communicator=false;
  this->_partition_spacing = new parameter_range_partition[nparam];
  for (int i=0; i<nparam; i++) this->_partition_spacing[i]=parameter_range_partition::GEOMETRIC;
  this->_partition_info = new int[nparam];
  for (int i=0; i<nparam; i++) this->_partition_info[i]=this->_partitions_per_dimension;
}

piecewise_hyperparameter_pack::piecewise_hyperparameter_pack(const piecewise_hyperparameter_pack& rhs) : hyperparameter_pack(rhs){
  this->_partitions_per_dimension=rhs._partitions_per_dimension;
  this->_observations_per_partition=rhs._observations_per_partition;
  this->_max_partition_spacing_factor=rhs._max_partition_spacing_factor;
  this->_partition_spacing = new parameter_range_partition[rhs._nparam];
  for (int i=0; i<rhs._nparam; i++) this->_partition_spacing[i]=rhs._partition_spacing[i];
  this->_partition_info = new int[rhs._nparam];
  for (int i=0; i<rhs._nparam; i++) this->_partition_info[i]=rhs._partition_info[i];
}

void piecewise_hyperparameter_pack::get(hyperparameter_pack& rhs) const{
  this->hyperparameter_pack::get(rhs);
  piecewise_hyperparameter_pack& rhs_derived = dynamic_cast<piecewise_hyperparameter_pack&>(rhs);
  assert(this->_nparam == rhs_derived._nparam);
  rhs_derived._partitions_per_dimension=this->_partitions_per_dimension;
  rhs_derived._observations_per_partition=this->_observations_per_partition;
  rhs_derived._max_partition_spacing_factor=this->_max_partition_spacing_factor;
  for (int i=0; i<rhs_derived._nparam; i++) rhs_derived._partition_spacing[i]=this->_partition_spacing[i];
  for (int i=0; i<rhs_derived._nparam; i++) rhs_derived._partition_info[i]=this->_partition_info[i];
}

void piecewise_hyperparameter_pack::set(const hyperparameter_pack& rhs){
  this->hyperparameter_pack::set(rhs);
  const piecewise_hyperparameter_pack& rhs_derived = dynamic_cast<const piecewise_hyperparameter_pack&>(rhs);
  assert(this->_nparam == rhs_derived._nparam);
  this->_partitions_per_dimension=rhs_derived._partitions_per_dimension;
  this->_observations_per_partition=rhs_derived._observations_per_partition;
  this->_max_partition_spacing_factor=rhs_derived._max_partition_spacing_factor;
  for (int i=0; i<rhs_derived._nparam; i++) this->_partition_spacing[i]=rhs_derived._partition_spacing[i];
  for (int i=0; i<rhs_derived._nparam; i++) this->_partition_info[i]=rhs_derived._partition_info[i];
}

piecewise_hyperparameter_pack::~piecewise_hyperparameter_pack(){
 delete[] this->_partition_info;
 delete[] this->_partition_spacing;
}

cpr_hyperparameter_pack::cpr_hyperparameter_pack(int nparam) : piecewise_hyperparameter_pack(nparam){
  // Default
  this->_partitions_per_dimension=16;
  this->_observations_per_partition=32;
  this->_cp_rank=3;
  this->_runtime_transformation = runtime_transformation::LOG;
  this->_parameter_transformation = parameter_transformation::LOG;
  this->_regularization=1e-4;
  this->_max_partition_spacing_factor=2;
  this->_max_num_re_inits = 10;
  this->_optimization_convergence_tolerance_for_re_init = 1e-1;
  this->_interpolation_factor_tolerance = 0.5;
  this->_max_num_optimization_sweeps=10;
  this->_optimization_convergence_tolerance=1e-2;
  this->_factor_matrix_optimization_max_num_iterations=10;
  this->_factor_matrix_optimization_convergence_tolerance=1e-2;
  this->_optimization_barrier_start=1e1;
  this->_optimization_barrier_stop=1e-11;
  this->_optimization_barrier_reduction_factor=8;
  this->_cm_training=MPI_COMM_SELF;
  this->_cm_data=MPI_COMM_SELF;
  this->_aggregate_obs_across_communicator=false;
  for (int i=0; i<nparam; i++) this->_partition_spacing[i]=parameter_range_partition::GEOMETRIC;
  for (int i=0; i<nparam; i++) this->_partition_info[i]=this->_partitions_per_dimension;
  this->_info = new double[7+nparam];
  for (int i=0; i<7+nparam; i++) this->_info[i]=-1; 
}

cpr_hyperparameter_pack::cpr_hyperparameter_pack(const cpr_hyperparameter_pack& rhs) : piecewise_hyperparameter_pack(rhs){
  this->_cp_rank=rhs._cp_rank;
  this->_regularization=rhs._regularization;
  this->_max_num_re_inits = rhs._max_num_re_inits;
  this->_optimization_convergence_tolerance_for_re_init = rhs._optimization_convergence_tolerance_for_re_init;
  this->_interpolation_factor_tolerance = rhs._interpolation_factor_tolerance;
  this->_max_num_optimization_sweeps=rhs._max_num_optimization_sweeps;
  this->_optimization_convergence_tolerance=rhs._optimization_convergence_tolerance;
  this->_factor_matrix_optimization_max_num_iterations=rhs._factor_matrix_optimization_max_num_iterations;
  this->_factor_matrix_optimization_convergence_tolerance=rhs._factor_matrix_optimization_convergence_tolerance;
  this->_optimization_barrier_start=rhs._optimization_barrier_start;
  this->_optimization_barrier_stop=rhs._optimization_barrier_stop;
  this->_optimization_barrier_reduction_factor=rhs._optimization_barrier_reduction_factor;
  this->_info = new double[7+rhs._nparam];
  for (int i=0; i<7+rhs._nparam; i++) this->_info[i]=rhs._info[i]; 
}

void cpr_hyperparameter_pack::get(hyperparameter_pack& rhs) const{
  this->piecewise_hyperparameter_pack::get(rhs);
  cpr_hyperparameter_pack& rhs_derived = dynamic_cast<cpr_hyperparameter_pack&>(rhs);
  assert(this->_nparam == rhs_derived._nparam);
  rhs_derived._cp_rank=this->_cp_rank;
  rhs_derived._regularization=this->_regularization;
  rhs_derived._max_num_re_inits = this->_max_num_re_inits;
  rhs_derived._optimization_convergence_tolerance_for_re_init = this->_optimization_convergence_tolerance_for_re_init;
  rhs_derived._interpolation_factor_tolerance = this->_interpolation_factor_tolerance;
  rhs_derived._max_num_optimization_sweeps=this->_max_num_optimization_sweeps;
  rhs_derived._optimization_convergence_tolerance=this->_optimization_convergence_tolerance;
  rhs_derived._factor_matrix_optimization_max_num_iterations=this->_factor_matrix_optimization_max_num_iterations;
  rhs_derived._factor_matrix_optimization_convergence_tolerance=this->_factor_matrix_optimization_convergence_tolerance;
  rhs_derived._optimization_barrier_start=this->_optimization_barrier_start;
  rhs_derived._optimization_barrier_stop=this->_optimization_barrier_stop;
  rhs_derived._optimization_barrier_reduction_factor=this->_optimization_barrier_reduction_factor;
  for (int i=0; i<7+rhs_derived._nparam; i++) rhs_derived._info[i]=this->_info[i]; 
}

void cpr_hyperparameter_pack::set(const hyperparameter_pack& rhs){
  this->piecewise_hyperparameter_pack::set(rhs);
  const cpr_hyperparameter_pack& rhs_derived = dynamic_cast<const cpr_hyperparameter_pack&>(rhs);
  assert(this->_nparam == rhs_derived._nparam);
  this->_cp_rank=rhs_derived._cp_rank;
  this->_regularization=rhs_derived._regularization;
  this->_max_num_re_inits = rhs_derived._max_num_re_inits;
  this->_optimization_convergence_tolerance_for_re_init = rhs_derived._optimization_convergence_tolerance_for_re_init;
  this->_interpolation_factor_tolerance = rhs_derived._interpolation_factor_tolerance;
  this->_max_num_optimization_sweeps=rhs_derived._max_num_optimization_sweeps;
  this->_optimization_convergence_tolerance=rhs_derived._optimization_convergence_tolerance;
  this->_factor_matrix_optimization_max_num_iterations=rhs_derived._factor_matrix_optimization_max_num_iterations;
  this->_factor_matrix_optimization_convergence_tolerance=rhs_derived._factor_matrix_optimization_convergence_tolerance;
  this->_optimization_barrier_start=rhs_derived._optimization_barrier_start;
  this->_optimization_barrier_stop=rhs_derived._optimization_barrier_stop;
  this->_optimization_barrier_reduction_factor=rhs_derived._optimization_barrier_reduction_factor;
  for (int i=0; i<7+rhs_derived._nparam; i++) this->_info[i]=rhs_derived._info[i]; 
}

cpr_hyperparameter_pack::~cpr_hyperparameter_pack(){
 delete[] this->_info;
}

cprg_hyperparameter_pack::cprg_hyperparameter_pack(int nparam) : cpr_hyperparameter_pack(nparam){
  // Default
  this->_max_spline_degree=1;
}

cprg_hyperparameter_pack::cprg_hyperparameter_pack(const cprg_hyperparameter_pack& rhs) : cpr_hyperparameter_pack(rhs){
  this->_max_spline_degree = rhs._max_spline_degree;
}

void cprg_hyperparameter_pack::get(hyperparameter_pack& rhs) const{
  this->cpr_hyperparameter_pack::get(rhs);
  cprg_hyperparameter_pack& rhs_derived = dynamic_cast<cprg_hyperparameter_pack&>(rhs);
  rhs_derived._max_spline_degree = this->_max_spline_degree;
}

void cprg_hyperparameter_pack::set(const hyperparameter_pack& rhs){
  this->cpr_hyperparameter_pack::set(rhs);
  const cprg_hyperparameter_pack& rhs_derived = dynamic_cast<const cprg_hyperparameter_pack&>(rhs);
  this->_max_spline_degree = rhs_derived._max_spline_degree;
}

cprg_hyperparameter_pack::~cprg_hyperparameter_pack(){}

};

