#include <cassert>
#include <fstream>

#include "hyperparameter_pack.h"
#include "types.h"

namespace performance_model{

hyperparameter_pack::hyperparameter_pack(int nparam){
  this->_runtime_transformation=runtime_transformation::LOG;
  this->_parameter_transformation=parameter_transformation::LOG;
  this->_loss_function=loss_function::MSE;
  this->_nparam = nparam;
  this->_cm_training = MPI_COMM_SELF;
  this->_cm_data = MPI_COMM_SELF;
  this->_aggregate_obs_across_communicator = false;
  this->_min_num_distinct_observed_configurations = 64;
}

hyperparameter_pack::hyperparameter_pack(const hyperparameter_pack& rhs){
  this->_runtime_transformation = rhs._runtime_transformation;
  this->_parameter_transformation = rhs._parameter_transformation;
  this->_loss_function = rhs._loss_function;
  this->_nparam = rhs._nparam;
  this->_cm_training = rhs._cm_training;
  this->_cm_data = rhs._cm_data;
  this->_aggregate_obs_across_communicator = rhs._aggregate_obs_across_communicator;
  this->_min_num_distinct_observed_configurations = rhs._min_num_distinct_observed_configurations;
}

void hyperparameter_pack::get(hyperparameter_pack& rhs) const{
  assert(this->_nparam == rhs._nparam);
  rhs._runtime_transformation = this->_runtime_transformation;
  rhs._parameter_transformation = this->_parameter_transformation;
  rhs._loss_function = this->_loss_function;
  rhs._nparam = this->_nparam;
  rhs._cm_training = this->_cm_training;
  rhs._cm_data = this->_cm_data;
  rhs._aggregate_obs_across_communicator = this->_aggregate_obs_across_communicator;
  rhs._min_num_distinct_observed_configurations = this->_min_num_distinct_observed_configurations;
}

void hyperparameter_pack::set(const hyperparameter_pack& rhs){
  assert(this->_nparam == rhs._nparam);
  this->_runtime_transformation = rhs._runtime_transformation;
  this->_parameter_transformation = rhs._parameter_transformation;
  this->_loss_function = rhs._loss_function;
  this->_nparam = rhs._nparam;
  this->_cm_training = rhs._cm_training;
  this->_cm_data = rhs._cm_data;
  this->_aggregate_obs_across_communicator = rhs._aggregate_obs_across_communicator;
  this->_min_num_distinct_observed_configurations = rhs._min_num_distinct_observed_configurations;
}

hyperparameter_pack::~hyperparameter_pack(){}

void hyperparameter_pack::write_to_file(std::ofstream& file) const{
  if (!file.is_open()) return;
  file << this->_nparam << "\n";
  if (this->_loss_function == loss_function::MSE){
    file << "MSE\n";
  } else if (this->_loss_function == loss_function::MLOGQ2){
    file << "MLOGQ2\n";
  } else file << "UNKNOWN\n";
  if (this->_runtime_transformation == runtime_transformation::NONE){
    file << "NONE\n";
  } else if (this->_runtime_transformation == runtime_transformation::LOG){
    file << "LOG\n";
  } else file << "UNKNOWN\n";
  if (this->_parameter_transformation == parameter_transformation::NONE){
    file << "NONE\n";
  } else if (this->_parameter_transformation == parameter_transformation::LOG){
    file << "LOG\n";
  } else file << "UNKNOWN\n";
  file << this->_aggregate_obs_across_communicator << "\n";
  file << this->_min_num_distinct_observed_configurations << "\n"; 
  //NOTE: _cm_training and _cm_data not written to file, nor read from file. Those must be reset by user, but will be set to defaults when reading in from file.
}

void hyperparameter_pack::read_from_file(std::ifstream& file){
  if (!file.is_open()) return;
  file >> this->_nparam;
  std::string temp;
  file >> temp;
  if (temp == "MSE"){
    this->_loss_function = loss_function::MSE;
  } else if (temp == "MLOGQ2"){
    this->_loss_function = loss_function::MLOGQ2;
  } else assert(0);
  file >> temp;
  if (temp == "NONE"){
    this->_runtime_transformation = runtime_transformation::NONE;
  } else if (temp == "LOG"){
    this->_runtime_transformation = runtime_transformation::LOG;
  } else assert(0);
  file >> temp;
  if (temp == "NONE"){
    this->_parameter_transformation = parameter_transformation::NONE;
  } else if (temp == "LOG"){
    this->_parameter_transformation = parameter_transformation::LOG;
  } else assert(0);
  file >> this->_aggregate_obs_across_communicator;
  file >> this->_min_num_distinct_observed_configurations; 
  //NOTE: _cm_training and _cm_data not written to file, nor read from file. Those must be reset by user, but will be set to defaults when reading in from file.
  this->_cm_data = MPI_COMM_SELF;
  this->_cm_training = MPI_COMM_SELF;
}

};
