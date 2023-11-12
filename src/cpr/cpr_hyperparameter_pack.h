#ifndef __PERFORMANCE_MODEL__CPR_HYPERPARAMETER_PACK_H_
#define __PERFORMANCE_MODEL__CPR_HYPERPARAMETER_PACK_H_

#include "../hyperparameter_pack.h"

namespace performance_model{

enum class parameter_range_partition;

class piecewise_hyperparameter_pack : public hyperparameter_pack{
public:
  piecewise_hyperparameter_pack(int nparam);
  piecewise_hyperparameter_pack(const piecewise_hyperparameter_pack& rhs);
  piecewise_hyperparameter_pack& operator=(const piecewise_hyperparameter_pack& rhs) = delete;
  virtual ~piecewise_hyperparameter_pack();

  virtual void get(hyperparameter_pack& rhs) const override;
  virtual void set(const hyperparameter_pack& rhs) override;
  virtual void write_to_file(std::ofstream& file) const;
  virtual void read_from_file(std::ifstream& file);

  int partitions_per_dimension;
  int observations_per_partition;
  int* partition_info;
  parameter_range_partition* partition_spacing;
  double max_partition_spacing_factor;
};

class cpr_hyperparameter_pack : public piecewise_hyperparameter_pack{
public:
  cpr_hyperparameter_pack(int nparam);
  cpr_hyperparameter_pack(const cpr_hyperparameter_pack& rhs);
  cpr_hyperparameter_pack& operator=(const cpr_hyperparameter_pack& rhs) = delete;
  virtual ~cpr_hyperparameter_pack();

  virtual void get(hyperparameter_pack& rhs) const override;
  virtual void set(const hyperparameter_pack& rhs) override;
  virtual void write_to_file(std::ofstream& file) const;
  virtual void read_from_file(std::ifstream& file);

  int cp_rank;
  double regularization;
  int max_num_re_inits;
  double optimization_convergence_tolerance_for_re_init;
  double interpolation_factor_tolerance;
  int max_num_optimization_sweeps;
  double optimization_convergence_tolerance;
  int factor_matrix_optimization_max_num_iterations;
  double factor_matrix_optimization_convergence_tolerance;
  double optimization_barrier_start;
  double optimization_barrier_stop;
  double optimization_barrier_reduction_factor;
};

class cprg_hyperparameter_pack : public cpr_hyperparameter_pack{
public:
  cprg_hyperparameter_pack(int nparam);
  cprg_hyperparameter_pack(const cprg_hyperparameter_pack& rhs);
  cprg_hyperparameter_pack& operator=(const cprg_hyperparameter_pack& rhs) = delete;
  virtual ~cprg_hyperparameter_pack();

  virtual void get(hyperparameter_pack& rhs) const override;
  virtual void set(const hyperparameter_pack& rhs) override;
  virtual void write_to_file(std::ofstream& file) const;
  virtual void read_from_file(std::ifstream& file);

  int max_spline_degree;
  runtime_transformation factor_matrix_element_transformation;
  parameter_transformation factor_matrix_underlying_position_transformation;
};

};
#endif // __PERFORMANCE_MODEL__CPR_HYPERPARAMETER_PACK_H_
