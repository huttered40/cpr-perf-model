#ifndef __PERFORMANCE_MODEL__MODEL_H_
#define __PERFORMANCE_MODEL__MODEL_H_

#include <vector>

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
  int num_distinct_configurations{0};
};

class model{
public:
  model(int nparam, const parameter_type* parameter_types, const hyperparameter_pack* pack);
  model(const model& rhs) = delete;
  model& operator=(const model& rhs) = delete;
  //TODO: What about std::vector<parameter_range_partition> interval_spacing? This is specific to CPR, not to all partitions
  model(const char* file_path);
  virtual ~model();
  virtual double predict(const double* configuration) const;
  virtual bool train(int& num_configurations, const double*& configurations, const double*& runtimes, bool save_dataset=false, model_fit_info* fit_info = nullptr) = 0;
  virtual void write_to_file(const char* file_path) const;
  virtual void read_from_file(const char* file_path);
  double get_min_observed_parameter_value(int parameter_id) const;
  double get_max_observed_parameter_value(int parameter_id) const;
  int get_num_inputs() const;
  virtual void get_hyperparameters(hyperparameter_pack& rhs) const;
  virtual void set_hyperparameters(const hyperparameter_pack& pack);
  virtual void get_parameters(parameter_pack& rhs) const;
  virtual void set_parameters(const parameter_pack& rhs);

protected:
  // Characteristics of the input data, NOT the model itself.
  int m_nparam;
  parameter_type* param_types;
  double* parameter_range_max;
  double* parameter_range_min;
  bool allocated_data;
  int num_distinct_configurations;

  hyperparameter_pack* hyperparameters;
  parameter_pack* parameters;
};

};
#endif // __PERFORMANCE_MODEL__MODEL_H_
