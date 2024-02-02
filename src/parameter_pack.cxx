#include <cassert>

#include "parameter_pack.h"

namespace performance_model{

parameter_pack::parameter_pack(){
}

parameter_pack::parameter_pack(const parameter_pack&){
}

parameter_pack::~parameter_pack(){}

void parameter_pack::get(parameter_pack&) const{
}

void parameter_pack::set(const parameter_pack&){
}

void parameter_pack::write_to_file(std::ofstream&) const{
}

void parameter_pack::read_from_file(std::ifstream&){
}

};
