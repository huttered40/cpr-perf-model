#ifndef __PERFORMANCE_MODEL__HYPERPARAMETER_PACK_H_
#define __PERFORMANCE_MODEL__HYPERPARAMETER_PACK_H_

#ifdef CRITTER
#include "critter_mpi.h"
#include "critter_symbol.h"
#else
#include <mpi.h>
#endif

namespace performance_model{

enum class runtime_transformation;
enum class parameter_transformation;
enum class loss_function;

class hyperparameter_pack{
public:
  hyperparameter_pack(int nparam);
  hyperparameter_pack(const hyperparameter_pack& rhs);
  virtual ~hyperparameter_pack();
  hyperparameter_pack& operator=(const hyperparameter_pack& rhs) = delete;
  virtual void get(hyperparameter_pack& rhs) const;
  virtual void set(const hyperparameter_pack& rhs);
  virtual void write_to_file(const char* file_path) const;
  virtual void read_from_file(const char* file_path);

  loss_function _loss_function;
  parameter_transformation _parameter_transformation;
  runtime_transformation _runtime_transformation;
  MPI_Comm _cm_training;
  MPI_Comm _cm_data;
  bool _aggregate_obs_across_communicator;
  int _nparam;
  int _min_num_distinct_observed_configurations;
};

};

#endif // __PERFORMANCE_MODEL__HYPERPARAMETER_PACK_H_
