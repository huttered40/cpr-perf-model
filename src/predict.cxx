#include <cassert>

#include "predict.h"
#include "model.h"

namespace performance_model{

double predict(const double* configuration, const model* interpolator, const model* extrapolator){
  int nparam = extrapolator->get_num_inputs();
  assert(nparam == extrapolator->get_num_inputs());
  bool use_interpolation_model = true;
  for (int i=0; i<nparam; i++){
    if (interpolator->get_max_observed_parameter_value(i) < configuration[i] || interpolator->get_min_observed_parameter_value(i) > configuration[i]){
      use_interpolation_model = false;
      break;
    }
  }
  return use_interpolation_model ? interpolator->predict(configuration) : extrapolator->predict(configuration);
}

};
