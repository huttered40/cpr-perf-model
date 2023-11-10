#ifndef __PERFORMANCE_MODEL__PARAMETER_PACK_H_
#define __PERFORMANCE_MODEL__PARAMETER_PACK_H_

namespace performance_model{

class parameter_pack{
public:
  parameter_pack();
  parameter_pack(const parameter_pack& rhs);
  virtual ~parameter_pack();
  parameter_pack& operator=(const parameter_pack& rhs) = delete;
  virtual void get(parameter_pack& rhs) const;
  virtual void set(const parameter_pack& rhs);
  virtual void write_to_file(const char* file) const;
  virtual void read_from_file(const char* file);
};

};

#endif // __PERFORMANCE_MODEL__PARAMETER_PACK_H_
