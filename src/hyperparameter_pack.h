#ifndef __PERFORMANCE_MODEL__HYPERPARAMETER_PACK_H_
#define __PERFORMANCE_MODEL__HYPERPARAMETER_PACK_H_

#ifdef CRITTER
#include "critter_mpi.h"
#include "critter_symbol.h"
#else
#include <mpi.h>
#endif

#include <fstream>

namespace performance_model{

enum class runtime_transformation;
enum class parameter_transformation;
enum class loss_function;

class hyperparameter_pack{
public:
  hyperparameter_pack(int t_nparam);
  hyperparameter_pack(const hyperparameter_pack& rhs);
  virtual ~hyperparameter_pack();
  hyperparameter_pack& operator=(const hyperparameter_pack& rhs) = delete;
  virtual void get(hyperparameter_pack& rhs) const;
  virtual void set(const hyperparameter_pack& rhs);
  virtual void write_to_file(std::ofstream& file_path) const;
  virtual void read_from_file(std::ifstream& file_path);

  loss_function loss;
  parameter_transformation parameter_transform;
  runtime_transformation runtime_transform;
  MPI_Comm cm_training;
  MPI_Comm cm_data;
  bool aggregate_obs_across_communicator;
  size_t nparam;
  size_t min_num_distinct_observed_configurations;
};

};

#endif // __PERFORMANCE_MODEL__HYPERPARAMETER_PACK_H_
