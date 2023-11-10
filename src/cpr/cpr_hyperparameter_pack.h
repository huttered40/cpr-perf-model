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

  int _partitions_per_dimension;
  int _observations_per_partition;
  int* _partition_info;
  parameter_range_partition* _partition_spacing;
  double _max_partition_spacing_factor;
};

class cpr_hyperparameter_pack : public piecewise_hyperparameter_pack{
public:
  cpr_hyperparameter_pack(int nparam);
  cpr_hyperparameter_pack(const cpr_hyperparameter_pack& rhs);
  cpr_hyperparameter_pack& operator=(const cpr_hyperparameter_pack& rhs) = delete;
  virtual ~cpr_hyperparameter_pack();

  virtual void get(hyperparameter_pack& rhs) const override;
  virtual void set(const hyperparameter_pack& rhs) override;

  int _cp_rank;
  double _regularization;
  int _max_num_re_inits;
  double _optimization_convergence_tolerance_for_re_init;
  double _interpolation_factor_tolerance;
  int _max_num_optimization_sweeps;
  double _optimization_convergence_tolerance;
  int _factor_matrix_optimization_max_num_iterations;
  double _factor_matrix_optimization_convergence_tolerance;
  double _optimization_barrier_start;
  double _optimization_barrier_stop;
  double _optimization_barrier_reduction_factor;
  double* _info;
};

class cprg_hyperparameter_pack : public cpr_hyperparameter_pack{
public:
  cprg_hyperparameter_pack(int nparam);
  cprg_hyperparameter_pack(const cprg_hyperparameter_pack& rhs);
  cprg_hyperparameter_pack& operator=(const cprg_hyperparameter_pack& rhs) = delete;
  virtual ~cprg_hyperparameter_pack();

  virtual void get(hyperparameter_pack& rhs) const override;
  virtual void set(const hyperparameter_pack& rhs) override;

  int _max_spline_degree;
};

};
#endif // __PERFORMANCE_MODEL__CPR_HYPERPARAMETER_PACK_H_
