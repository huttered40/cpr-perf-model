#ifndef __PERFORMANCE_MODEL__CPR_PARAMETER_PACK_H_
#define __PERFORMANCE_MODEL__CPR_PARAMETER_PACK_H_

#include <vector>

#include "../parameter_pack.h"

namespace performance_model{

class parameter_pack;

// Handles trivial knots for categorical parameters and numerical parameters with no subdomain
class piecewise_parameter_pack : public parameter_pack{
public:
  piecewise_parameter_pack();
  piecewise_parameter_pack(const piecewise_parameter_pack& rhs);
  virtual ~piecewise_parameter_pack();
  piecewise_parameter_pack& operator=(const piecewise_parameter_pack& rhs) = delete;
  void get(parameter_pack& rhs) const override;
  void set(const parameter_pack& rhs) override;

  std::vector<int> num_partitions_per_dimension;
  std::vector<double> knot_positions;
  std::vector<int> knot_index_offsets;
};

class tensor_parameter_pack : public piecewise_parameter_pack{
public:
  tensor_parameter_pack();
  tensor_parameter_pack(const tensor_parameter_pack& rhs);
  virtual ~tensor_parameter_pack();
  tensor_parameter_pack& operator=(const tensor_parameter_pack& rhs) = delete;
  void get(parameter_pack& rhs) const override;
  void set(const parameter_pack& rhs) override;

  double* tensor_elements;
};

class cpr_parameter_pack : public piecewise_parameter_pack{
public:
  cpr_parameter_pack();
  cpr_parameter_pack(const cpr_parameter_pack& rhs);
  virtual ~cpr_parameter_pack();
  cpr_parameter_pack& operator=(const cpr_parameter_pack& rhs) = delete;
  void get(parameter_pack& rhs) const override;
  void set(const parameter_pack& rhs) override;

  int cp_rank;
  // mode-lengths per mode of tensor can be attained via num_partitions_per_dimension
  double* factor_matrix_elements;
};

class cprg_parameter_pack : public cpr_parameter_pack{
public:
  cprg_parameter_pack();
  cprg_parameter_pack(const cprg_parameter_pack& rhs);
  virtual ~cprg_parameter_pack();
  cprg_parameter_pack& operator=(const cprg_parameter_pack& rhs) = delete;
  void get(parameter_pack& rhs) const override;
  void set(const parameter_pack& rhs) override;

  int spline_degree;
  double* global_models;
};

};

#endif // __PERFORMANCE_MODEL__CPR_PARAMETER_PACK_H_
