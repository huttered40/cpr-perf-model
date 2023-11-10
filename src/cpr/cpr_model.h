#ifndef __PERFORMANCE_MODEL__CPR_MODEL_H_
#define __PERFORMANCE_MODEL__CPR_MODEL_H_

#include <vector>//TODO: Remove later

#include "../model.h"

namespace performance_model{

enum class parameter_type;
enum class parameter_range_partition;

class hyperparameter_pack;
class cpr_hyperparameter_pack;
class cprg_hyperparameter_pack;
class cpr_parameter_pack;
class cprg_parameter_pack;

class tensor_model_fit_info : public model_fit_info{
public:
  tensor_model_fit_info();
  tensor_model_fit_info(const tensor_model_fit_info& rhs);
  tensor_model_fit_info& operator=(const tensor_model_fit_info& rhs);
  virtual ~tensor_model_fit_info();
  double tensor_density{-1};
  double num_tensor_elements{-1};
  double quadrature_error{-1};
};

class cpr_model_fit_info : public tensor_model_fit_info{
public:
  cpr_model_fit_info();
  cpr_model_fit_info(const cpr_model_fit_info& rhs);
  cpr_model_fit_info& operator=(const cpr_model_fit_info& rhs);
  virtual ~cpr_model_fit_info();
  double loss{-1};
  double low_rank_approximation_error{-1};
};

class cprg_model_fit_info : public cpr_model_fit_info{
public:
  cprg_model_fit_info();
  cprg_model_fit_info(const cprg_model_fit_info& rhs);
  cprg_model_fit_info& operator=(const cprg_model_fit_info& rhs);
  virtual ~cprg_model_fit_info();
};

class cpr_model : public model{
public:
  cpr_model(int nparam, const parameter_type* parameter_types, const hyperparameter_pack* pack=nullptr);
  cpr_model(const cpr_model& rhs) = delete;
  cpr_model& operator=(const cpr_model& rhs) = delete;
  virtual ~cpr_model() override;
  virtual double predict(const double* configuration) const override;
  virtual bool train(int& num_configurations, const double*& configurations, const double*& runtimes, bool save_dataset=false, model_fit_info* fit_info=nullptr) override;
  virtual void write_to_file(const char* file_path) const override;
  virtual void read_from_file(const char* file_path) override;
  void get_hyperparameters(hyperparameter_pack& pack) const override;
  void set_hyperparameters(const hyperparameter_pack& pack) override;
  virtual void get_parameters(parameter_pack& rhs) const override;
  virtual void set_parameters(const parameter_pack& rhs) override;

protected:
  void write_to_file(std::ofstream& file) const;
  void read_from_file(std::ifstream& file);
  void init(std::vector<int>& cells_info, const std::vector<double> custom_grid_pts={},
    int num_configurations=-1, const double* features=nullptr);

  bool m_is_valid;
  int order;
  std::vector<int> numerical_modes;
  std::vector<int> categorical_modes;
  std::vector<std::vector<int>> Projected_Omegas;
};

class cprg_model : public cpr_model{
public:
  cprg_model(int nparam, const parameter_type* parameter_types, const hyperparameter_pack* pack=nullptr);
  cprg_model(const cprg_model& rhs) = delete;
  cprg_model& operator=(const cprg_model& rhs) = delete;
  virtual ~cprg_model() override;
  virtual double predict(const double* configuration) const override;
  virtual bool train(int& num_configurations, const double*& configurations, const double*& runtimes, bool save_dataset=false, model_fit_info* fit_info=nullptr) override;
  virtual void write_to_file(const char* file_path) const override;
  virtual void read_from_file(const char* file_path) override;
  void get_hyperparameters(hyperparameter_pack& pack) const override;
  void set_hyperparameters(const hyperparameter_pack& pack) override;
  virtual void get_parameters(parameter_pack& rhs) const override;
  virtual void set_parameters(const parameter_pack& rhs) override;

protected:
  void write_to_file(std::ofstream& file) const;
  void read_from_file(std::ifstream& file);
};
};
#endif // __PERFORMANCE_MODEL__CPR_HYPERPARAMETER_PACK_H_
