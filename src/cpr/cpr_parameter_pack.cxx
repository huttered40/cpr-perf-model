#include <cassert>
#include <cstring>

#include "cpr_parameter_pack.h"

namespace performance_model{

piecewise_parameter_pack::piecewise_parameter_pack() : parameter_pack(){
  this->num_partitions_per_dimension.clear();
  this->knot_positions.clear();
  this->knot_index_offsets.clear();
  this->num_dimensions = 0;
  this->num_knots = 0;
}

piecewise_parameter_pack::piecewise_parameter_pack(const piecewise_parameter_pack& rhs) : parameter_pack(rhs){
  this->num_partitions_per_dimension = rhs.num_partitions_per_dimension;
  this->knot_positions = rhs.knot_positions;
  this->knot_index_offsets = rhs.knot_index_offsets;
  this->num_dimensions = rhs.num_dimensions;
  this->num_knots = rhs.num_knots;
  assert(this->num_knots == this->knot_positions.size());
  assert(this->num_dimensions == this->knot_index_offsets.size());
  assert(this->num_dimensions == this->num_partitions_per_dimension.size());
}

piecewise_parameter_pack::~piecewise_parameter_pack(){
  this->num_partitions_per_dimension.clear();
  this->knot_positions.clear();
  this->knot_index_offsets.clear();
  this->num_dimensions = 0;
  this->num_knots = 0;
}

void piecewise_parameter_pack::get(parameter_pack& rhs) const{
  this->parameter_pack::get(rhs);
  piecewise_parameter_pack& rhs_derived = dynamic_cast<piecewise_parameter_pack&>(rhs);
  rhs_derived.num_partitions_per_dimension = this->num_partitions_per_dimension;
  rhs_derived.knot_positions = this->knot_positions;
  rhs_derived.knot_index_offsets = this->knot_index_offsets;
  rhs_derived.num_dimensions = this->num_dimensions;
  rhs_derived.num_knots = this->num_knots;
}

void piecewise_parameter_pack::set(const parameter_pack& rhs){
  this->parameter_pack::set(rhs);
  const piecewise_parameter_pack& rhs_derived = dynamic_cast<const piecewise_parameter_pack&>(rhs);
  this->num_partitions_per_dimension = rhs_derived.num_partitions_per_dimension;
  this->knot_positions = rhs_derived.knot_positions;
  this->knot_index_offsets = rhs_derived.knot_index_offsets;
  this->num_dimensions = rhs_derived.num_dimensions;
  this->num_knots = rhs_derived.num_knots;
}

void piecewise_parameter_pack::write_to_file(std::ofstream& file) const{
  if (!file.is_open()) return;
  this->parameter_pack::write_to_file(file);
  file << this->num_dimensions << "\n";
  file << this->num_knots << "\n";
  for (int i=0; i<num_partitions_per_dimension.size(); i++){
    if (i>0) file << ",";
    file << num_partitions_per_dimension[i];
  } file << "\n";
  for (int i=0; i<knot_index_offsets.size(); i++){
    if (i>0) file << ",";
    file << knot_index_offsets[i];
  } file << "\n";
  for (int i=0; i<knot_positions.size(); i++){
    if (i>0) file << ",";
    file << knot_positions[i];
  } file << "\n";
}

void piecewise_parameter_pack::read_from_file(std::ifstream& file){
  if (!file.is_open()) return;
  this->parameter_pack::read_from_file(file);
  file >> this->num_dimensions;
  file >> this->num_knots;
  this->num_partitions_per_dimension.resize(this->num_dimensions);
  this->knot_index_offsets.resize(this->num_dimensions);
  this->knot_positions.resize(this->num_knots);
  std::string temp;
  for (int i=0; i<num_partitions_per_dimension.size(); i++){
    if (i==(num_partitions_per_dimension.size()-1)) getline(file,temp,'\n');
    else getline(file,temp,',');
    this->num_partitions_per_dimension[i] = std::stoi(temp);
  }
  for (int i=0; i<knot_index_offsets.size(); i++){
    if (i==(knot_index_offsets.size()-1)) getline(file,temp,'\n');
    else getline(file,temp,',');
    this->knot_index_offsets[i] = std::stoi(temp);
  }
  for (int i=0; i<knot_positions.size(); i++){
    if (i==(knot_positions.size()-1)) getline(file,temp,'\n');
    else getline(file,temp,',');
    this->knot_positions[i] = std::stod(temp);
  }
}

tensor_parameter_pack::tensor_parameter_pack() : piecewise_parameter_pack(){
  this->tensor_elements = nullptr;
  this->num_tensor_elements = 0;
}

tensor_parameter_pack::tensor_parameter_pack(const tensor_parameter_pack& rhs) : piecewise_parameter_pack(rhs){
  int nelements_rhs = 0;
  for (const auto& it : rhs.num_partitions_per_dimension) nelements_rhs *= it;
  assert(rhs.num_tensor_elements == nelements_rhs);
  this->tensor_elements = new double[nelements_rhs];
  std::memcpy(this->tensor_elements,rhs.tensor_elements,nelements_rhs*sizeof(double));
}

tensor_parameter_pack::~tensor_parameter_pack(){
  if (this->tensor_elements != nullptr) delete[] this->tensor_elements;
  this->num_tensor_elements = 0;
}

void tensor_parameter_pack::get(parameter_pack& rhs) const{
  this->piecewise_parameter_pack::get(rhs);
  tensor_parameter_pack& rhs_derived = dynamic_cast<tensor_parameter_pack&>(rhs);
  // Assume that rhs has no dynamically-allocated memory in rhs.tensor_elements. User can delete it
  rhs_derived.tensor_elements = new double[this->num_tensor_elements];
  std::memcpy(rhs_derived.tensor_elements,this->tensor_elements,this->num_tensor_elements*sizeof(double));
}

void tensor_parameter_pack::set(const parameter_pack& rhs){
  this->piecewise_parameter_pack::set(rhs);
  const tensor_parameter_pack& rhs_derived = dynamic_cast<const tensor_parameter_pack&>(rhs);
  if (this->tensor_elements != nullptr) delete[] this->tensor_elements;
  int nelements_rhs = 0;
  for (const auto& it : rhs_derived.num_partitions_per_dimension) nelements_rhs *= it;
  assert(nelements_rhs == rhs_derived.num_tensor_elements);
  this->tensor_elements = new double[nelements_rhs];
  std::memcpy(this->tensor_elements,rhs_derived.tensor_elements,nelements_rhs*sizeof(double));
  this->num_tensor_elements = rhs_derived.num_tensor_elements;
}

void tensor_parameter_pack::write_to_file(std::ofstream& file) const{
  if (!file.is_open()) return;
  this->piecewise_parameter_pack::write_to_file(file);
  if (this->tensor_elements == nullptr) return;
  file << this->num_tensor_elements << "\n";
  for (int64_t i=0; i<this->num_tensor_elements; i++){
    if (i>0) file << ",";
    file << this->tensor_elements[i];
  } file << "\n";
}

void tensor_parameter_pack::read_from_file(std::ifstream& file){
  if (!file.is_open()) return;
  this->piecewise_parameter_pack::read_from_file(file);
  file >> this->num_tensor_elements;
  if (this->tensor_elements == nullptr) this->tensor_elements = new double[this->num_tensor_elements];
  std::string temp;
  for (int i=0; i<this->num_tensor_elements; i++){
    if (i==(this->num_tensor_elements-1)) getline(file,temp,'\n');
    else getline(file,temp,',');
    this->tensor_elements[i] = std::stod(temp);
  }
}


cpr_parameter_pack::cpr_parameter_pack() : piecewise_parameter_pack(){
  this->factor_matrix_elements = nullptr;
  this->cp_rank = 0;
  this->num_factor_matrix_elements = 0;
}

cpr_parameter_pack::cpr_parameter_pack(const cpr_parameter_pack& rhs) : piecewise_parameter_pack(rhs){
  int nelements_rhs = 0;
  for (const auto& it : rhs.num_partitions_per_dimension) nelements_rhs += it;
  nelements_rhs *= rhs.cp_rank;
  assert(nelements_rhs == rhs.num_factor_matrix_elements);
  if (this->factor_matrix_elements == nullptr){
    this->factor_matrix_elements = new double[nelements_rhs];
  }
  std::memcpy(this->factor_matrix_elements,rhs.factor_matrix_elements,nelements_rhs*sizeof(double));
  this->num_factor_matrix_elements = rhs.num_factor_matrix_elements;
}

cpr_parameter_pack::~cpr_parameter_pack(){
  if (this->factor_matrix_elements != nullptr) delete[] this->factor_matrix_elements;
  this->factor_matrix_elements=nullptr;
  this->num_factor_matrix_elements = 0;
  this->cp_rank = 0;
}

void cpr_parameter_pack::get(parameter_pack& rhs) const{
  this->piecewise_parameter_pack::get(rhs);
  cpr_parameter_pack& rhs_derived = dynamic_cast<cpr_parameter_pack&>(rhs);
  rhs_derived.factor_matrix_elements = new double[this->num_factor_matrix_elements];
  std::memcpy(rhs_derived.factor_matrix_elements,this->factor_matrix_elements,this->num_factor_matrix_elements*sizeof(double));
  rhs_derived.cp_rank = this->cp_rank;
  rhs_derived.num_factor_matrix_elements = this->num_factor_matrix_elements;
}

void cpr_parameter_pack::set(const parameter_pack& rhs){
  this->piecewise_parameter_pack::set(rhs);
  const cpr_parameter_pack& rhs_derived = dynamic_cast<const cpr_parameter_pack&>(rhs);
  if (this->factor_matrix_elements != nullptr) delete[] this->factor_matrix_elements;
  int nelements_rhs = 0;
  for (const auto& it : rhs_derived.num_partitions_per_dimension) nelements_rhs += it;
  nelements_rhs *= rhs_derived.cp_rank;
  assert(nelements_rhs == rhs_derived.num_factor_matrix_elements);
  this->factor_matrix_elements = new double[nelements_rhs];
  std::memcpy(this->factor_matrix_elements,rhs_derived.factor_matrix_elements,nelements_rhs*sizeof(double));
  this->cp_rank = rhs_derived.cp_rank;
  this->num_factor_matrix_elements = rhs_derived.num_factor_matrix_elements;
}

void cpr_parameter_pack::write_to_file(std::ofstream& file) const{
  if (!file.is_open()) return;
  this->piecewise_parameter_pack::write_to_file(file);
  file << this->cp_rank << "\n";
  file << this->num_factor_matrix_elements << "\n";
  for (int i=0; i<this->num_factor_matrix_elements; i++){
    if (i>0) file << ",";
    file << this->factor_matrix_elements[i];
  } file << "\n";
}

void cpr_parameter_pack::read_from_file(std::ifstream& file){
  if (!file.is_open()) return;
  this->piecewise_parameter_pack::read_from_file(file);
  file >> this->cp_rank;
  file >> this->num_factor_matrix_elements;
  if (this->factor_matrix_elements == nullptr) this->factor_matrix_elements = new double[this->num_factor_matrix_elements];
  std::string temp;
  for (int i=0; i<this->num_factor_matrix_elements; i++){
    if (i==(this->num_factor_matrix_elements-1)) getline(file,temp,'\n');
    else getline(file,temp,',');
    this->factor_matrix_elements[i] = std::stod(temp);
  }
}


cprg_parameter_pack::cprg_parameter_pack() : cpr_parameter_pack(){
  this->global_models = nullptr;
  this->num_models = 0;
  this->spline_degree = 0;
}

cprg_parameter_pack::cprg_parameter_pack(const cprg_parameter_pack& rhs) : cpr_parameter_pack(rhs){
  int nelements_rhs = rhs.num_models*(2+rhs.spline_degree+rhs.cp_rank);
  this->global_models = new double[nelements_rhs];
  std::memcpy(this->global_models,rhs.global_models,nelements_rhs*sizeof(double));
  this->spline_degree = rhs.spline_degree;
  this->num_models = rhs.num_models;
}

cprg_parameter_pack::~cprg_parameter_pack(){
  if (this->global_models != nullptr) delete[] this->global_models;
  this->global_models=nullptr;
  this->spline_degree = 0;
  this->num_models = 0;
}

void cprg_parameter_pack::get(parameter_pack& rhs) const{
  this->cpr_parameter_pack::get(rhs);
  cprg_parameter_pack& rhs_derived = dynamic_cast<cprg_parameter_pack&>(rhs);
  int nelements = this->num_models*(2+this->spline_degree+this->cp_rank);
  rhs_derived.global_models = new double[nelements];
  std::memcpy(rhs_derived.global_models,this->global_models,nelements*sizeof(double));
  rhs_derived.spline_degree = this->spline_degree;
  rhs_derived.num_models = this->num_models;
}

void cprg_parameter_pack::set(const parameter_pack& rhs){
  this->cpr_parameter_pack::set(rhs);
  const cprg_parameter_pack& rhs_derived = dynamic_cast<const cprg_parameter_pack&>(rhs);
  int nelements_rhs = rhs_derived.num_models*(2+rhs_derived.spline_degree+rhs_derived.cp_rank);
  this->global_models = new double[nelements_rhs];
  std::memcpy(this->global_models,rhs_derived.global_models,nelements_rhs*sizeof(double));
  this->spline_degree = rhs_derived.spline_degree;
  this->num_models = rhs_derived.num_models;
}

void cprg_parameter_pack::write_to_file(std::ofstream& file) const{
  if (!file.is_open()) return;
  this->cpr_parameter_pack::write_to_file(file);
  file << this->spline_degree;
  file << this->num_models; 
  int nelements = this->num_models*(2+this->spline_degree+this->cp_rank);
  for (int i=0; i<nelements; i++){
    if (i>0) file << ",";
    file << this->global_models[i];
  } file << "\n";
}

void cprg_parameter_pack::read_from_file(std::ifstream& file){
  if (!file.is_open()) return;
  this->cpr_parameter_pack::read_from_file(file);
  file >> this->spline_degree;
  file >> this->num_models; 
  int nelements = this->num_models*(2+this->spline_degree+this->cp_rank);
  if (this->global_models == nullptr) this->global_models = new double[nelements];
  std::string temp;
  for (int i=0; i<nelements; i++){
    if (i==(nelements-1)) getline(file,temp,'\n');
    else getline(file,temp,',');
    this->global_models[i] = std::stod(temp);
  }
}

};
