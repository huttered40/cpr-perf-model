#include <cassert>
#include <climits>

#include "cpr_hyperparameter_pack.h"
#include "cpr_types.h"
#include "../types.h"

namespace performance_model{

piecewise_hyperparameter_pack::piecewise_hyperparameter_pack(size_t nparam) : hyperparameter_pack(nparam){
  // Default
  this->partitions_per_dimension=8;
  this->observations_per_partition=32;
  this->runtime_transform = runtime_transformation::NONE;
  this->parameter_transform = parameter_transformation::NONE;
  this->max_partition_spacing_factor=2;
  this->cm_training=MPI_COMM_SELF;
  this->cm_data=MPI_COMM_SELF;
  this->aggregate_obs_across_communicator=false;
  this->partition_spacing = new parameter_range_partition[nparam];
  for (size_t i=0; i<nparam; i++) this->partition_spacing[i]=parameter_range_partition::GEOMETRIC;
  this->partition_info = new int[nparam];
  for (size_t i=0; i<nparam; i++) this->partition_info[i]=this->partitions_per_dimension;
}

piecewise_hyperparameter_pack::piecewise_hyperparameter_pack(const piecewise_hyperparameter_pack& rhs) : hyperparameter_pack(rhs){
  this->partitions_per_dimension=rhs.partitions_per_dimension;
  this->observations_per_partition=rhs.observations_per_partition;
  this->max_partition_spacing_factor=rhs.max_partition_spacing_factor;
  this->partition_spacing = new parameter_range_partition[rhs.nparam];
  for (size_t i=0; i<rhs.nparam; i++) this->partition_spacing[i]=rhs.partition_spacing[i];
  this->partition_info = new int[rhs.nparam];
  for (size_t i=0; i<rhs.nparam; i++) this->partition_info[i]=rhs.partition_info[i];
}

void piecewise_hyperparameter_pack::get(hyperparameter_pack& rhs) const{
  this->hyperparameter_pack::get(rhs);
  piecewise_hyperparameter_pack& rhs_derived = dynamic_cast<piecewise_hyperparameter_pack&>(rhs);
  assert(this->nparam == rhs_derived.nparam);
  rhs_derived.partitions_per_dimension=this->partitions_per_dimension;
  rhs_derived.observations_per_partition=this->observations_per_partition;
  rhs_derived.max_partition_spacing_factor=this->max_partition_spacing_factor;
  for (size_t i=0; i<rhs_derived.nparam; i++) rhs_derived.partition_spacing[i]=this->partition_spacing[i];
  for (size_t i=0; i<rhs_derived.nparam; i++) rhs_derived.partition_info[i]=this->partition_info[i];
}

void piecewise_hyperparameter_pack::set(const hyperparameter_pack& rhs){
  this->hyperparameter_pack::set(rhs);
  const piecewise_hyperparameter_pack& rhs_derived = dynamic_cast<const piecewise_hyperparameter_pack&>(rhs);
  assert(this->nparam == rhs_derived.nparam);
  this->partitions_per_dimension=rhs_derived.partitions_per_dimension;
  this->observations_per_partition=rhs_derived.observations_per_partition;
  this->max_partition_spacing_factor=rhs_derived.max_partition_spacing_factor;
  for (size_t i=0; i<rhs_derived.nparam; i++) this->partition_spacing[i]=rhs_derived.partition_spacing[i];
  for (size_t i=0; i<rhs_derived.nparam; i++) this->partition_info[i]=rhs_derived.partition_info[i];
}

piecewise_hyperparameter_pack::~piecewise_hyperparameter_pack(){
 delete[] this->partition_info;
 delete[] this->partition_spacing;
}

void piecewise_hyperparameter_pack::write_to_file(std::ofstream& file) const{
  if (!file.is_open()) return;
  this->hyperparameter_pack::write_to_file(file);
  file << this->partitions_per_dimension << "\n";
  file << this->observations_per_partition << "\n";
  for (size_t i=0; i<this->nparam; i++){
    if (i>0) file << ",";
    file << this->partition_info[i];
  } file << "\n";
  for (size_t i=0; i<this->nparam; i++){
    if (i>0) file << ",";
    if (partition_spacing[i]==parameter_range_partition::SINGLE) file << "SINGLE";
    else if (partition_spacing[i]==parameter_range_partition::AUTOMATIC) file << "AUTOMATIC";
    else if (partition_spacing[i]==parameter_range_partition::UNIFORM) file << "UNIFORM";
    else if (partition_spacing[i]==parameter_range_partition::GEOMETRIC) file << "GEOMETRIC";
    else if (partition_spacing[i]==parameter_range_partition::CUSTOM) file << "CUSTOM";
  } file << "\n";
  file << this->max_partition_spacing_factor << "\n";
}

void piecewise_hyperparameter_pack::read_from_file(std::ifstream& file){
  if (!file.is_open()) return;
  this->hyperparameter_pack::read_from_file(file);
  file >> this->partitions_per_dimension;
  file >> this->observations_per_partition;
  if (this->partition_info == nullptr) this->partition_info = new int[this->nparam];
  if (this->partition_spacing == nullptr) this->partition_spacing = new parameter_range_partition[this->nparam];
  std::string temp;
  for (size_t i=0; i<this->nparam; i++){
    if (i==(this->nparam-1)) getline(file,temp,'\n');
    else getline(file,temp,',');
    this->partition_info[i] = std::stoi(temp);
  }
  for (size_t i=0; i<this->nparam; i++){
    if (i==(this->nparam-1)) getline(file,temp,'\n');
    else getline(file,temp,',');
    if (temp == "SINGLE") partition_spacing[i] =parameter_range_partition::SINGLE;
    else if (temp == "AUTOMATIC") partition_spacing[i] = parameter_range_partition::AUTOMATIC;
    else if (temp == "UNIFORM") partition_spacing[i] = parameter_range_partition::UNIFORM;
    else if (temp == "GEOMETRIC") partition_spacing[i] = parameter_range_partition::GEOMETRIC;
    else if (temp == "CUSTOM") partition_spacing[i] = parameter_range_partition::CUSTOM;
  }
  file >> this->max_partition_spacing_factor;
}

cpr_hyperparameter_pack::cpr_hyperparameter_pack(size_t nparam) : piecewise_hyperparameter_pack(nparam){
  // Default
  this->partitions_per_dimension=16;
  this->observations_per_partition=32;
  this->cp_rank=3;
  this->runtime_transform = runtime_transformation::LOG;
  this->parameter_transform = parameter_transformation::LOG;
  this->regularization=1e-4;
  this->max_partition_spacing_factor=2;
  this->max_num_re_inits = 10;
  this->optimization_convergence_tolerance_for_re_init = 1e-1;
  this->interpolation_factor_tolerance = 0.5;
  this->max_num_optimization_sweeps=10;
  this->optimization_convergence_tolerance=1e-2;
  this->factor_matrix_optimization_max_num_iterations=10;
  this->factor_matrix_optimization_convergence_tolerance=1e-2;
  this->optimization_barrier_start=1e1;
  this->optimization_barrier_stop=1e-11;
  this->optimization_barrier_reduction_factor=8;
  this->cm_training=MPI_COMM_SELF;
  this->cm_data=MPI_COMM_SELF;
  this->aggregate_obs_across_communicator=false;
  for (size_t i=0; i<nparam; i++) this->partition_spacing[i]=parameter_range_partition::GEOMETRIC;
  for (size_t i=0; i<nparam; i++) this->partition_info[i]=this->partitions_per_dimension;
}

cpr_hyperparameter_pack::cpr_hyperparameter_pack(const cpr_hyperparameter_pack& rhs) : piecewise_hyperparameter_pack(rhs){
  this->cp_rank=rhs.cp_rank;
  this->regularization=rhs.regularization;
  this->max_num_re_inits = rhs.max_num_re_inits;
  this->optimization_convergence_tolerance_for_re_init = rhs.optimization_convergence_tolerance_for_re_init;
  this->interpolation_factor_tolerance = rhs.interpolation_factor_tolerance;
  this->max_num_optimization_sweeps=rhs.max_num_optimization_sweeps;
  this->optimization_convergence_tolerance=rhs.optimization_convergence_tolerance;
  this->factor_matrix_optimization_max_num_iterations=rhs.factor_matrix_optimization_max_num_iterations;
  this->factor_matrix_optimization_convergence_tolerance=rhs.factor_matrix_optimization_convergence_tolerance;
  this->optimization_barrier_start=rhs.optimization_barrier_start;
  this->optimization_barrier_stop=rhs.optimization_barrier_stop;
  this->optimization_barrier_reduction_factor=rhs.optimization_barrier_reduction_factor;
}

void cpr_hyperparameter_pack::get(hyperparameter_pack& rhs) const{
  this->piecewise_hyperparameter_pack::get(rhs);
  cpr_hyperparameter_pack& rhs_derived = dynamic_cast<cpr_hyperparameter_pack&>(rhs);
  assert(this->nparam == rhs_derived.nparam);
  rhs_derived.cp_rank=this->cp_rank;
  rhs_derived.regularization=this->regularization;
  rhs_derived.max_num_re_inits = this->max_num_re_inits;
  rhs_derived.optimization_convergence_tolerance_for_re_init = this->optimization_convergence_tolerance_for_re_init;
  rhs_derived.interpolation_factor_tolerance = this->interpolation_factor_tolerance;
  rhs_derived.max_num_optimization_sweeps=this->max_num_optimization_sweeps;
  rhs_derived.optimization_convergence_tolerance=this->optimization_convergence_tolerance;
  rhs_derived.factor_matrix_optimization_max_num_iterations=this->factor_matrix_optimization_max_num_iterations;
  rhs_derived.factor_matrix_optimization_convergence_tolerance=this->factor_matrix_optimization_convergence_tolerance;
  rhs_derived.optimization_barrier_start=this->optimization_barrier_start;
  rhs_derived.optimization_barrier_stop=this->optimization_barrier_stop;
  rhs_derived.optimization_barrier_reduction_factor=this->optimization_barrier_reduction_factor;
}

void cpr_hyperparameter_pack::set(const hyperparameter_pack& rhs){
  this->piecewise_hyperparameter_pack::set(rhs);
  const cpr_hyperparameter_pack& rhs_derived = dynamic_cast<const cpr_hyperparameter_pack&>(rhs);
  assert(this->nparam == rhs_derived.nparam);
  this->cp_rank=rhs_derived.cp_rank;
  this->regularization=rhs_derived.regularization;
  this->max_num_re_inits = rhs_derived.max_num_re_inits;
  this->optimization_convergence_tolerance_for_re_init = rhs_derived.optimization_convergence_tolerance_for_re_init;
  this->interpolation_factor_tolerance = rhs_derived.interpolation_factor_tolerance;
  this->max_num_optimization_sweeps=rhs_derived.max_num_optimization_sweeps;
  this->optimization_convergence_tolerance=rhs_derived.optimization_convergence_tolerance;
  this->factor_matrix_optimization_max_num_iterations=rhs_derived.factor_matrix_optimization_max_num_iterations;
  this->factor_matrix_optimization_convergence_tolerance=rhs_derived.factor_matrix_optimization_convergence_tolerance;
  this->optimization_barrier_start=rhs_derived.optimization_barrier_start;
  this->optimization_barrier_stop=rhs_derived.optimization_barrier_stop;
  this->optimization_barrier_reduction_factor=rhs_derived.optimization_barrier_reduction_factor;
}

cpr_hyperparameter_pack::~cpr_hyperparameter_pack(){
}

void cpr_hyperparameter_pack::write_to_file(std::ofstream& file) const{
  if (!file.is_open()) return;
  this->piecewise_hyperparameter_pack::write_to_file(file);
  file << this->cp_rank << "\n";
  file << this->regularization << "\n";
  file << this->max_num_re_inits << "\n";
  file << this->optimization_convergence_tolerance_for_re_init << "\n";
  file << this->interpolation_factor_tolerance << "\n";
  file << this->max_num_optimization_sweeps << "\n";
  file << this->optimization_convergence_tolerance << "\n";
  file << this->factor_matrix_optimization_max_num_iterations << "\n";
  file << this->factor_matrix_optimization_convergence_tolerance << "\n";
  file << this->optimization_barrier_start << "\n";
  file << this->optimization_barrier_stop << "\n";
  file << this->optimization_barrier_reduction_factor << "\n";
}

void cpr_hyperparameter_pack::read_from_file(std::ifstream& file){
  if (!file.is_open()) return;
  this->piecewise_hyperparameter_pack::read_from_file(file);
  file >> this->cp_rank;
  file >> this->regularization;
  file >> this->max_num_re_inits;
  file >> this->optimization_convergence_tolerance_for_re_init;
  file >> this->interpolation_factor_tolerance;
  file >> this->max_num_optimization_sweeps;
  file >> this->optimization_convergence_tolerance;
  file >> this->factor_matrix_optimization_max_num_iterations;
  file >> this->factor_matrix_optimization_convergence_tolerance;
  file >> this->optimization_barrier_start;
  file >> this->optimization_barrier_stop;
  file >> this->optimization_barrier_reduction_factor;
}

cprg_hyperparameter_pack::cprg_hyperparameter_pack(size_t nparam) : cpr_hyperparameter_pack(nparam){
  // Default
  this->max_spline_degree=1;
  this->max_training_set_size=INT_MAX;
  this->factor_matrix_element_transformation = runtime_transformation::LOG;
  this->factor_matrix_underlying_position_transformation = parameter_transformation::LOG;
}

cprg_hyperparameter_pack::cprg_hyperparameter_pack(const cprg_hyperparameter_pack& rhs) : cpr_hyperparameter_pack(rhs){
  this->max_spline_degree = rhs.max_spline_degree;
  this->max_training_set_size = rhs.max_training_set_size;
  this->factor_matrix_element_transformation = rhs.factor_matrix_element_transformation;
  this->factor_matrix_underlying_position_transformation = rhs.factor_matrix_underlying_position_transformation;
}

void cprg_hyperparameter_pack::get(hyperparameter_pack& rhs) const{
  this->cpr_hyperparameter_pack::get(rhs);
  cprg_hyperparameter_pack& rhs_derived = dynamic_cast<cprg_hyperparameter_pack&>(rhs);
  rhs_derived.max_spline_degree = this->max_spline_degree;
  rhs_derived.max_training_set_size = this->max_training_set_size;
  rhs_derived.factor_matrix_element_transformation = this->factor_matrix_element_transformation;
  rhs_derived.factor_matrix_underlying_position_transformation = this->factor_matrix_underlying_position_transformation;
}

void cprg_hyperparameter_pack::set(const hyperparameter_pack& rhs){
  this->cpr_hyperparameter_pack::set(rhs);
  const cprg_hyperparameter_pack& rhs_derived = dynamic_cast<const cprg_hyperparameter_pack&>(rhs);
  this->max_spline_degree = rhs_derived.max_spline_degree;
  this->max_training_set_size = rhs_derived.max_training_set_size;
  this->factor_matrix_element_transformation = rhs_derived.factor_matrix_element_transformation;
  this->factor_matrix_underlying_position_transformation = rhs_derived.factor_matrix_underlying_position_transformation;
}

cprg_hyperparameter_pack::~cprg_hyperparameter_pack(){}

void cprg_hyperparameter_pack::write_to_file(std::ofstream& file) const{
  if (!file.is_open()) return;
  this->cpr_hyperparameter_pack::write_to_file(file);
  file << this->max_spline_degree << "\n";
  file << this->max_training_set_size << "\n";
  if (this->factor_matrix_element_transformation == runtime_transformation::NONE){
    file << "NONE\n";
  } else if (this->factor_matrix_element_transformation == runtime_transformation::LOG){
    file << "LOG\n";
  }
  if (this->factor_matrix_underlying_position_transformation == parameter_transformation::NONE){
    file << "NONE\n";
  } else if (this->factor_matrix_underlying_position_transformation == parameter_transformation::LOG){
    file << "LOG\n";
  }
}

void cprg_hyperparameter_pack::read_from_file(std::ifstream& file){
  if (!file.is_open()) return;
  this->cpr_hyperparameter_pack::read_from_file(file);
  file >> this->max_spline_degree;
  file >> this->max_training_set_size;
  std::string temp;
  file >> temp;
  if (temp == "NONE"){
    this->factor_matrix_element_transformation = runtime_transformation::NONE;
  } else if (temp == "LOG"){
    this->factor_matrix_element_transformation = runtime_transformation::LOG;
  }
  file >> temp;
  if (temp == "NONE"){
    this->factor_matrix_underlying_position_transformation = parameter_transformation::NONE;
  } else if (temp == "LOG"){
    this->factor_matrix_underlying_position_transformation = parameter_transformation::LOG;
  }
}

};

