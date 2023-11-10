#ifndef __PERFORMANCE_MODEL__PREDICT_H_
#define __PERFORMANCE_MODEL__PREDICT_H_

namespace performance_model{

class model;

double predict(const double* configuration, const model* interpolator, const model* extrapolator);

};
#endif // __PERFORMANCE_MODEL__PREDICT_H_
