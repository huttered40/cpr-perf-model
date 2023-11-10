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

class cpr_model : public model{
public:
  cpr_model(int nparam, const parameter_type* parameter_types, const hyperparameter_pack* pack=nullptr);
  cpr_model(const char* file_path);
  cpr_model(const cpr_model& rhs) = delete;
  cpr_model& operator=(const cpr_model& rhs) = delete;
  virtual ~cpr_model() override;
  virtual double predict(const double* configuration) const override;
  virtual bool train(int& num_configurations, const double*& configurations, const double*& runtimes, bool compute_fit_error=true, bool save_dataset=false) override;
  virtual void write_to_file(const char* file_path) const override;
  virtual void read_from_file(const char* file_path) override;
  void get_hyperparameters(hyperparameter_pack& pack) const override;
  void set_hyperparameters(const hyperparameter_pack& pack) override;
  virtual void get_parameters(parameter_pack& rhs) const override;
  virtual void set_parameters(const parameter_pack& rhs) override;

protected:
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
  cprg_model(const char* file_path);
  cprg_model(const cprg_model& rhs) = delete;
  cprg_model& operator=(const cprg_model& rhs) = delete;
  virtual ~cprg_model() override;
  virtual double predict(const double* configuration) const override;
  virtual bool train(int& num_configurations, const double*& configurations, const double*& runtimes, bool compute_fit_error=true, bool save_dataset=false) override;
  virtual void write_to_file(const char* file_path) const override;
  virtual void read_from_file(const char* file_path) override;
  void get_hyperparameters(hyperparameter_pack& pack) const override;
  void set_hyperparameters(const hyperparameter_pack& pack) override;
  virtual void get_parameters(parameter_pack& rhs) const override;
  virtual void set_parameters(const parameter_pack& rhs) override;

};
};
#endif // __PERFORMANCE_MODEL__CPR_HYPERPARAMETER_PACK_H_
