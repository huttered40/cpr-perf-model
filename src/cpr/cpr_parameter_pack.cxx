#include <cassert>
#include <cstring>

#include "cpr_parameter_pack.h"

namespace performance_model{

piecewise_parameter_pack::piecewise_parameter_pack() : parameter_pack(){
  this->num_partitions_per_dimension.clear();
  this->knot_positions.clear();
  this->knot_index_offsets.clear();
}

piecewise_parameter_pack::piecewise_parameter_pack(const piecewise_parameter_pack& rhs) : parameter_pack(rhs){
  this->num_partitions_per_dimension = rhs.num_partitions_per_dimension;
  this->knot_positions = rhs.knot_positions;
  this->knot_index_offsets = rhs.knot_index_offsets;
}

piecewise_parameter_pack::~piecewise_parameter_pack(){
  this->num_partitions_per_dimension.clear();
  this->knot_positions.clear();
  this->knot_index_offsets.clear();
}

void piecewise_parameter_pack::get(parameter_pack& rhs) const{
  this->parameter_pack::get(rhs);
  piecewise_parameter_pack& rhs_derived = dynamic_cast<piecewise_parameter_pack&>(rhs);
  rhs_derived.num_partitions_per_dimension = this->num_partitions_per_dimension;
  rhs_derived.knot_positions = this->knot_positions;
  rhs_derived.knot_index_offsets = this->knot_index_offsets;
}

void piecewise_parameter_pack::set(const parameter_pack& rhs){
  this->parameter_pack::set(rhs);
  const piecewise_parameter_pack& rhs_derived = dynamic_cast<const piecewise_parameter_pack&>(rhs);
  this->num_partitions_per_dimension = rhs_derived.num_partitions_per_dimension;
  this->knot_positions = rhs_derived.knot_positions;
  this->knot_index_offsets = rhs_derived.knot_index_offsets;
}

tensor_parameter_pack::tensor_parameter_pack() : piecewise_parameter_pack(){
  this->tensor_elements = nullptr;
}

tensor_parameter_pack::tensor_parameter_pack(const tensor_parameter_pack& rhs) : piecewise_parameter_pack(rhs){
  int nelements_rhs = 0;
  for (const auto& it : rhs.num_partitions_per_dimension) nelements_rhs *= it;
  this->tensor_elements = new double[nelements_rhs];
  std::memcpy(this->tensor_elements,rhs.tensor_elements,nelements_rhs*sizeof(double));
}

tensor_parameter_pack::~tensor_parameter_pack(){
  if (this->tensor_elements != nullptr) delete[] this->tensor_elements;
}

void tensor_parameter_pack::get(parameter_pack& rhs) const{
  this->piecewise_parameter_pack::get(rhs);
  tensor_parameter_pack& rhs_derived = dynamic_cast<tensor_parameter_pack&>(rhs);
  // Assume that rhs has no dynamically-allocated memory in rhs.tensor_elements. User can delete it
  int nelements = 0;
  for (const auto& it : this->num_partitions_per_dimension) nelements *= it;
  rhs_derived.tensor_elements = new double[nelements];
  std::memcpy(rhs_derived.tensor_elements,this->tensor_elements,nelements*sizeof(double));
}

void tensor_parameter_pack::set(const parameter_pack& rhs){
  this->piecewise_parameter_pack::set(rhs);
  const tensor_parameter_pack& rhs_derived = dynamic_cast<const tensor_parameter_pack&>(rhs);
  if (this->tensor_elements != nullptr) delete[] this->tensor_elements;
  int nelements_rhs = 0;
  for (const auto& it : rhs_derived.num_partitions_per_dimension) nelements_rhs *= it;
  this->tensor_elements = new double[nelements_rhs];
  std::memcpy(this->tensor_elements,rhs_derived.tensor_elements,nelements_rhs*sizeof(double));
}


cpr_parameter_pack::cpr_parameter_pack() : piecewise_parameter_pack(){
  this->factor_matrix_elements = nullptr;
  this->cp_rank = 0;
}

cpr_parameter_pack::cpr_parameter_pack(const cpr_parameter_pack& rhs) : piecewise_parameter_pack(rhs){
  int nelements_rhs = 0;
  for (const auto& it : rhs.num_partitions_per_dimension) nelements_rhs += it;
  nelements_rhs *= rhs.cp_rank;
  if (this->factor_matrix_elements == nullptr){
    this->factor_matrix_elements = new double[nelements_rhs];
  }
  std::memcpy(this->factor_matrix_elements,rhs.factor_matrix_elements,nelements_rhs*sizeof(double));
}

cpr_parameter_pack::~cpr_parameter_pack(){
  if (this->factor_matrix_elements != nullptr) delete[] this->factor_matrix_elements;
  this->factor_matrix_elements=nullptr;
}

void cpr_parameter_pack::get(parameter_pack& rhs) const{
  this->piecewise_parameter_pack::get(rhs);
  cpr_parameter_pack& rhs_derived = dynamic_cast<cpr_parameter_pack&>(rhs);
  int nelements = 0;
  for (const auto& it : this->num_partitions_per_dimension) nelements += it;
  nelements *= this->cp_rank;
  rhs_derived.factor_matrix_elements = new double[nelements];
  std::memcpy(rhs_derived.factor_matrix_elements,this->factor_matrix_elements,nelements*sizeof(double));
  rhs_derived.cp_rank = this->cp_rank;
}

void cpr_parameter_pack::set(const parameter_pack& rhs){
  this->piecewise_parameter_pack::set(rhs);
  const cpr_parameter_pack& rhs_derived = dynamic_cast<const cpr_parameter_pack&>(rhs);
  if (this->factor_matrix_elements != nullptr) delete[] this->factor_matrix_elements;
  int nelements_rhs = 0;
  for (const auto& it : rhs_derived.num_partitions_per_dimension) nelements_rhs += it;
  nelements_rhs *= rhs_derived.cp_rank;
  this->factor_matrix_elements = new double[nelements_rhs];
  std::memcpy(this->factor_matrix_elements,rhs_derived.factor_matrix_elements,nelements_rhs*sizeof(double));
  this->cp_rank = rhs_derived.cp_rank;
}


cprg_parameter_pack::cprg_parameter_pack() : cpr_parameter_pack(){
  this->global_models = nullptr;
  this->spline_degree = 0;
}

cprg_parameter_pack::cprg_parameter_pack(const cprg_parameter_pack& rhs) : cpr_parameter_pack(rhs){
  int nelements_rhs = rhs.num_partitions_per_dimension.size()*(2+rhs.spline_degree+rhs.cp_rank);
  this->global_models = new double[nelements_rhs];
  std::memcpy(this->global_models,rhs.global_models,nelements_rhs*sizeof(double));
  this->spline_degree = rhs.spline_degree;
}

cprg_parameter_pack::~cprg_parameter_pack(){
  if (this->global_models != nullptr) delete[] this->global_models;
  this->global_models=nullptr;
}

void cprg_parameter_pack::get(parameter_pack& rhs) const{
  this->cpr_parameter_pack::get(rhs);
  cprg_parameter_pack& rhs_derived = dynamic_cast<cprg_parameter_pack&>(rhs);
  int nelements = this->num_partitions_per_dimension.size()*(2+this->spline_degree+this->cp_rank);
  rhs_derived.global_models = new double[nelements];
  std::memcpy(rhs_derived.global_models,this->global_models,nelements*sizeof(double));
  rhs_derived.spline_degree = this->spline_degree;
}

void cprg_parameter_pack::set(const parameter_pack& rhs){
  this->cpr_parameter_pack::set(rhs);
  const cprg_parameter_pack& rhs_derived = dynamic_cast<const cprg_parameter_pack&>(rhs);
  int nelements_rhs = rhs_derived.num_partitions_per_dimension.size()*(2+rhs_derived.spline_degree+rhs_derived.cp_rank);
  this->global_models = new double[nelements_rhs];
  std::memcpy(this->global_models,rhs_derived.global_models,nelements_rhs*sizeof(double));
  this->spline_degree = rhs_derived.spline_degree;
}

};
