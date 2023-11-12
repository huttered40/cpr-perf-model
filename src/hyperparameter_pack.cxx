#include <cassert>
#include <fstream>

#include "hyperparameter_pack.h"
#include "types.h"

namespace performance_model{

hyperparameter_pack::hyperparameter_pack(int t_nparam){
  this->runtime_transform=runtime_transformation::LOG;
  this->parameter_transform=parameter_transformation::LOG;
  this->loss=loss_function::MSE;
  this->nparam = t_nparam;
  this->cm_training = MPI_COMM_SELF;
  this->cm_data = MPI_COMM_SELF;
  this->aggregate_obs_across_communicator = false;
  this->min_num_distinct_observed_configurations = 64;
}

hyperparameter_pack::hyperparameter_pack(const hyperparameter_pack& rhs){
  this->runtime_transform = rhs.runtime_transform;
  this->parameter_transform = rhs.parameter_transform;
  this->loss = rhs.loss;
  this->nparam = rhs.nparam;
  this->cm_training = rhs.cm_training;
  this->cm_data = rhs.cm_data;
  this->aggregate_obs_across_communicator = rhs.aggregate_obs_across_communicator;
  this->min_num_distinct_observed_configurations = rhs.min_num_distinct_observed_configurations;
}

void hyperparameter_pack::get(hyperparameter_pack& rhs) const{
  assert(this->nparam == rhs.nparam);
  rhs.runtime_transform = this->runtime_transform;
  rhs.parameter_transform = this->parameter_transform;
  rhs.loss = this->loss;
  rhs.nparam = this->nparam;
  rhs.cm_training = this->cm_training;
  rhs.cm_data = this->cm_data;
  rhs.aggregate_obs_across_communicator = this->aggregate_obs_across_communicator;
  rhs.min_num_distinct_observed_configurations = this->min_num_distinct_observed_configurations;
}

void hyperparameter_pack::set(const hyperparameter_pack& rhs){
  assert(this->nparam == rhs.nparam);
  this->runtime_transform = rhs.runtime_transform;
  this->parameter_transform = rhs.parameter_transform;
  this->loss = rhs.loss;
  this->nparam = rhs.nparam;
  this->cm_training = rhs.cm_training;
  this->cm_data = rhs.cm_data;
  this->aggregate_obs_across_communicator = rhs.aggregate_obs_across_communicator;
  this->min_num_distinct_observed_configurations = rhs.min_num_distinct_observed_configurations;
}

hyperparameter_pack::~hyperparameter_pack(){}

void hyperparameter_pack::write_to_file(std::ofstream& file) const{
  if (!file.is_open()) return;
  file << this->nparam << "\n";
  if (this->loss == loss_function::MSE){
    file << "MSE\n";
  } else if (this->loss == loss_function::MLOGQ2){
    file << "MLOGQ2\n";
  } else file << "UNKNOWN\n";
  if (this->runtime_transform == runtime_transformation::NONE){
    file << "NONE\n";
  } else if (this->runtime_transform == runtime_transformation::LOG){
    file << "LOG\n";
  } else file << "UNKNOWN\n";
  if (this->parameter_transform == parameter_transformation::NONE){
    file << "NONE\n";
  } else if (this->parameter_transform == parameter_transformation::LOG){
    file << "LOG\n";
  } else file << "UNKNOWN\n";
  file << this->aggregate_obs_across_communicator << "\n";
  file << this->min_num_distinct_observed_configurations << "\n"; 
  //NOTE: cm_training and cm_data not written to file, nor read from file. Those must be reset by user, but will be set to defaults when reading in from file.
}

void hyperparameter_pack::read_from_file(std::ifstream& file){
  if (!file.is_open()) return;
  file >> this->nparam;
  std::string temp;
  file >> temp;
  if (temp == "MSE"){
    this->loss = loss_function::MSE;
  } else if (temp == "MLOGQ2"){
    this->loss = loss_function::MLOGQ2;
  } else assert(0);
  file >> temp;
  if (temp == "NONE"){
    this->runtime_transform = runtime_transformation::NONE;
  } else if (temp == "LOG"){
    this->runtime_transform = runtime_transformation::LOG;
  } else assert(0);
  file >> temp;
  if (temp == "NONE"){
    this->parameter_transform = parameter_transformation::NONE;
  } else if (temp == "LOG"){
    this->parameter_transform = parameter_transformation::LOG;
  } else assert(0);
  file >> this->aggregate_obs_across_communicator;
  file >> this->min_num_distinct_observed_configurations; 
  //NOTE: cm_training and cm_data not written to file, nor read from file. Those must be reset by user, but will be set to defaults when reading in from file.
  this->cm_data = MPI_COMM_SELF;
  this->cm_training = MPI_COMM_SELF;
}

};
