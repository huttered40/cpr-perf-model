#ifndef __PERFORMANCE_MODEL__MODEL_H_
#define __PERFORMANCE_MODEL__MODEL_H_

#include <vector>
#include <fstream>

// See types.h and cpr/cpr_types.h for values each enum can take
// Relevant enums (forward-declared below): parameter_type,
//                                          runtime_transformation,
//                                          parameter_transformation,
//                                          loss_function,
//                                          parameter_range_partition
//
// See cpr/hyperparameter_pack.h for available hyperparameters
//
// Use interface below to set/get both hyperparameters and parameters of model

namespace performance_model{

enum class parameter_type;

class hyperparameter_pack;
class parameter_pack;

class model_fit_info{
public:
  model_fit_info();
  model_fit_info(const model_fit_info& rhs);
  model_fit_info& operator=(const model_fit_info& rhs);
  virtual ~model_fit_info();
  double training_error{-1};
  size_t num_distinct_configurations{0};
};

class model{
public:
  model(size_t nparam, const parameter_type* parameter_types, const hyperparameter_pack* pack=nullptr);
  model(const model& rhs) = delete;
  model& operator=(const model& rhs) = delete;
  virtual ~model();
  virtual double predict(const double*) const;
  virtual bool train(size_t& num_configurations, const double*& configurations, const double*& runtimes, model_fit_info* fit_info = nullptr) = 0;
  virtual void write_to_file(const char* file_path) const;
  virtual void read_from_file(const char* file_path);
  double get_min_observed_parameter_value(int parameter_id) const;
  double get_max_observed_parameter_value(int parameter_id) const;
  size_t get_num_inputs() const;
  virtual void get_hyperparameters(hyperparameter_pack&) const;
  virtual void set_hyperparameters(const hyperparameter_pack&);
  virtual void get_parameters(parameter_pack& rhs) const;
  virtual void set_parameters(const parameter_pack& rhs);

protected:
  void write_to_file(std::ofstream& file) const;
  void read_from_file(std::ifstream& file);

  // Characteristics of the input data, NOT the model itself.
  size_t nparam;
  parameter_type* param_types;
  double* param_range_max;
  double* param_range_min;
  bool allocated_data;
  size_t num_distinct_configurations;

  hyperparameter_pack* hyperparameters;
  parameter_pack* parameters;
};

};
#endif // __PERFORMANCE_MODEL__MODEL_H_
