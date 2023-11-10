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

void hyperparameter_pack::write_to_file(const char* file_path) const{
  std::ofstream model_file_ptr;
  model_file_ptr.open(file_path,std::ios_base::app);
  if(model_file_ptr.fail()) return;
  if (this->_loss_function == loss_function::MSE){
    model_file_ptr << "MSE\n";
  } else if (this->_loss_function == loss_function::MLOGQ2){
    model_file_ptr << "MLOGQ2\n";
  } else model_file_ptr << "UNKNOWN\n";
  if (this->_runtime_transformation == runtime_transformation::NONE){
    model_file_ptr << "NONE\n";
  } else if (this->_runtime_transformation == runtime_transformation::LOG){
    model_file_ptr << "LOG\n";
  } else model_file_ptr << "UNKNOWN\n";
  if (this->_parameter_transformation == parameter_transformation::NONE){
    model_file_ptr << "NONE\n";
  } else if (this->_parameter_transformation == parameter_transformation::LOG){
    model_file_ptr << "LOG\n";
  } else model_file_ptr << "UNKNOWN\n";
  model_file_ptr.close();
}

void hyperparameter_pack::read_from_file(const char* file_path){
  std::ifstream model_file_ptr;
  model_file_ptr.open(file_path,std::ios_base::app);
  if(model_file_ptr.fail()) return;
  //TODO: Add reverse of above
  assert(0);
}

};
