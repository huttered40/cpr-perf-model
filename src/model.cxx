#ifdef CRITTER
#include "critter_mpi.h"
#include "critter_symbol.h"
#else
#include <mpi.h>
#endif


#include <cassert>
#include <cfloat>
#include <fstream>
#include <set>
#include <vector>
#include "model.h"
#include "types.h"
#include "hyperparameter_pack.h"
#include "parameter_pack.h"

namespace performance_model{

static bool aggregate_observations(int& num_configurations, const double*& configurations, const double*& runtimes, int order, MPI_Comm cm){
  // Return value indicates whether configurations and runtimes need to be explicitly deleted
  int rank,size,lnc;
  int internal_tag=0;
  int internal_tag1=0;
  int internal_tag2=0;
  MPI_Comm_rank(cm,&rank);
  MPI_Comm_size(cm,&size);
  if (size==1) return false;
  size_t active_size = size;
  size_t active_rank = rank;
  size_t active_mult = 1;
  int local_num_configurations = num_configurations;
  std::vector<double> local_configurations(num_configurations*order);
  std::vector<double> local_runtimes(num_configurations);
  for (int i=0; i<num_configurations; i++){
    for (int j=0; j<order; j++){
      local_configurations[i*order+j]=configurations[i*order+j];
    }
    local_runtimes[i]=runtimes[i];
  }
  while (active_size>1){
    if (active_rank % 2 == 1){
      int partner = (active_rank-1)*active_mult;
      // Send sizes before true message so that receiver can be aware of the array sizes for subsequent communication
      PMPI_Send(&local_num_configurations,1,MPI_INT,partner,internal_tag,cm);
      if (local_num_configurations>0){
        // Send active kernels with keys
        PMPI_Send(&local_configurations[0],local_num_configurations*order,MPI_DOUBLE,partner,internal_tag1,cm);
        PMPI_Send(&local_runtimes[0],local_num_configurations,MPI_DOUBLE,partner,internal_tag2,cm);
      }
      break;// important. Senders must not update {active_size,active_rank,active_mult}
    }
    else if ((active_rank % 2 == 0) && (active_rank < (active_size-1))){
      int partner = (active_rank+1)*active_mult;
      // Recv sizes of arrays to create buffers for subsequent communication. Goal is to concatenate, not to replace
      PMPI_Recv(&lnc,1,MPI_INT,partner,internal_tag,cm,MPI_STATUS_IGNORE);
      if (lnc>0){
        local_num_configurations += lnc;
        local_configurations.resize(local_num_configurations*order);
        local_runtimes.resize(local_num_configurations);
        PMPI_Recv(&local_configurations[order*(local_num_configurations-lnc)],lnc*order,MPI_DOUBLE,partner,internal_tag1,cm,MPI_STATUS_IGNORE);
        PMPI_Recv(&local_runtimes[local_num_configurations-lnc],lnc,MPI_DOUBLE,partner,internal_tag2,cm,MPI_STATUS_IGNORE);
      }
    }
    active_size = active_size/2 + active_size%2;
    active_rank /= 2;
    active_mult *= 2;
  }
  // Goal is to replace, not to concatenate
  PMPI_Bcast(&local_num_configurations,1,MPI_INT,0,cm);
  assert(num_configurations <= local_num_configurations);
  if (local_num_configurations == 0){
    return false;
  }
  if (rank != 0){
    local_configurations.resize(local_num_configurations*order);
    local_runtimes.resize(local_num_configurations);
  }
  PMPI_Bcast(&local_configurations[0],local_num_configurations*order,MPI_DOUBLE,0,cm);
  PMPI_Bcast(&local_runtimes[0],local_num_configurations,MPI_DOUBLE,0,cm);
  double* _runtimes = new double[local_num_configurations];
  double* _configurations = new double[local_num_configurations*order];
  for (int i=0; i<local_num_configurations; i++){
    for (int j=0; j<order; j++){
      _configurations[i*order+j]=local_configurations[i*order+j];
    }
    _runtimes[i]=local_runtimes[i];
  }
  configurations = _configurations;
  runtimes = _runtimes;
  num_configurations=local_num_configurations;
  return true;
}

model_fit_info::model_fit_info(){}
model_fit_info::model_fit_info(const model_fit_info& rhs){
  this->training_error = rhs.training_error;
  this->num_distinct_configurations = rhs.num_distinct_configurations;
}
model_fit_info& model_fit_info::operator=(const model_fit_info& rhs){
  this->training_error = rhs.training_error;
  this->num_distinct_configurations = rhs.num_distinct_configurations;
}
model_fit_info::~model_fit_info(){
}

model::model(int nparam, const parameter_type* parameter_types, const hyperparameter_pack* pack){
  this->hyperparameters = new hyperparameter_pack(*pack);
  this->parameters = new parameter_pack();
  this->m_nparam = nparam;
  this->allocated_data = false;
  this->num_distinct_configurations = 0;
  this->param_types = new parameter_type[nparam];
  this->parameter_range_min = new double[nparam];
  this->parameter_range_max = new double[nparam];
  for (int i=0; i<nparam; i++){
    this->param_types[i] = parameter_types[i];
    this->parameter_range_min[i] = DBL_MAX;
    this->parameter_range_max[i] = DBL_MIN;
  }
}
model::model(const char* file_path){
  //TODO
}
double model::predict(const double* configuration) const{
  assert(0);
  return 0;
}
model::~model(){
  if (this->hyperparameters != nullptr) delete this->hyperparameters;
  if (this->parameters != nullptr) delete this->parameters;
  if (this->param_types != nullptr) delete[] this->param_types;
  if (this->parameter_range_min != nullptr) delete[] this->parameter_range_min;
  if (this->parameter_range_max != nullptr) delete[] this->parameter_range_max;
  this->hyperparameters = nullptr;
  this->parameters = nullptr;
  this->param_types = nullptr;
  this->parameter_range_min = nullptr;
  this->parameter_range_max = nullptr;
}
bool model::train(int& num_configurations, const double*& configurations, const double*& runtimes, bool save_dataset, model_fit_info* fit_info){
  int world_size_for_training,world_size_for_data_aggregation;
  MPI_Comm_size(this->hyperparameters->_cm_training,&world_size_for_training);
  MPI_Comm_size(this->hyperparameters->_cm_data,&world_size_for_data_aggregation);

  const double* save_c_ptr = configurations;
  const double* save_r_ptr = runtimes;

  this->allocated_data = false;
  if (this->hyperparameters->_aggregate_obs_across_communicator){
    this->allocated_data = aggregate_observations(num_configurations,configurations,runtimes,this->m_nparam,this->hyperparameters->_cm_data);
    if (this->allocated_data){
      assert(configurations != save_c_ptr);
      assert(runtimes != save_r_ptr);
    } else{
      if (num_configurations>0){
	assert(configurations == save_c_ptr);
	assert(runtimes == save_r_ptr);
      }
    }
  }

  //TODO: Average into sample mean here, rather than waiting for later, as we do not need to find a node_key yet!
  // Check how many distinct configurations there are.
  std::set<std::vector<double>> distinct_configuration_dict;
  std::vector<double> config(this->m_nparam);
  for (int i=0; i<num_configurations; i++){
    for (int j=0; j<this->m_nparam; j++){
      config[j]=configurations[i*this->m_nparam+j];
    }
    distinct_configuration_dict.insert(config);
  }
  this->num_distinct_configurations = distinct_configuration_dict.size();
  if (fit_info != nullptr) fit_info->num_distinct_configurations = this->num_distinct_configurations;
  // No point in training if the sample size is so small
  // We only really care about distinct configurations
  if (distinct_configuration_dict.size() < this->hyperparameters->_min_num_distinct_observed_configurations){
    if (this->allocated_data){
      delete[] configurations;
      delete[] runtimes;
      this->allocated_data = false;
    }
    return false;
  }
  distinct_configuration_dict.clear();// TODO: Later don't do this

  for (int i=0; i<num_configurations; i++){
    for (int j=0; j<this->m_nparam; j++){
      this->parameter_range_min[j]=std::min(this->parameter_range_min[j],configurations[i*this->m_nparam+j]);
      this->parameter_range_max[j]=std::max(this->parameter_range_max[j],configurations[i*this->m_nparam+j]);
    }
  }
  for (int j=0; j<this->m_nparam; j++){
    if (this->parameter_range_max[j] < this->parameter_range_min[j]){
      if (this->allocated_data){
        delete[] configurations;
        delete[] runtimes;
      }
      return false;
    }
  }
  return true;
}
void model::write_to_file(const char* file_path) const{
  std::ofstream model_file_ptr;
  model_file_ptr.open(file_path,std::ios_base::app);
  if(model_file_ptr.fail()) return;
  model_file_ptr << this->m_nparam << "\n";
  for (int i=0; i<this->m_nparam; i++){
    if (i>0) model_file_ptr << ",";
    if (param_types[i] == parameter_type::NUMERICAL) model_file_ptr << "NUMERICAL";
    else model_file_ptr << "CATEGORICAL";
  } model_file_ptr << "\n";
  for (int i=0; i<this->m_nparam; i++){
    if (i>0) model_file_ptr << ",";
    model_file_ptr << parameter_range_min[i] << ",";
  } model_file_ptr << "\n";
  for (int i=0; i<this->m_nparam; i++){
    if (i>0) model_file_ptr << ",";
    model_file_ptr << parameter_range_max[i] << ",";
  } model_file_ptr << "\n";
  model_file_ptr.close();
}
void model::read_from_file(const char* file_path){
  std::ifstream model_file_ptr;
  model_file_ptr.open(file_path,std::ios_base::out);
  //TODO: Add reverse of above
  assert(0);
}
double model::get_min_observed_parameter_value(int parameter_id) const{
  return this->parameter_range_min[parameter_id];
}
double model::get_max_observed_parameter_value(int parameter_id) const{
  return this->parameter_range_max[parameter_id];
}
void model::get_hyperparameters(hyperparameter_pack& rhs) const{
  assert(0);
}
void model::set_hyperparameters(const hyperparameter_pack& pack){
  assert(0);
}
int model::get_num_inputs() const{
  return this->m_nparam;
}
void model::get_parameters(parameter_pack& rhs) const{
  this->parameters->get(rhs);
}
void model::set_parameters(const parameter_pack& rhs){
  this->parameters->set(rhs);
}

};
