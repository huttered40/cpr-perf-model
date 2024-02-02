#ifndef __PERFORMANCE_MODEL__PARAMETER_PACK_H_
#define __PERFORMANCE_MODEL__PARAMETER_PACK_H_

#include <fstream>

namespace performance_model{

class parameter_pack{
public:
  parameter_pack();
  parameter_pack(const parameter_pack&);
  virtual ~parameter_pack();
  parameter_pack& operator=(const parameter_pack&) = delete;
  virtual void get(parameter_pack&) const;
  virtual void set(const parameter_pack&);
  virtual void write_to_file(std::ofstream&) const;
  virtual void read_from_file(std::ifstream&);
};

};

#endif // __PERFORMANCE_MODEL__PARAMETER_PACK_H_
