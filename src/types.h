#ifndef __PERFORMANCE_MODEL__TYPES_H_
#define __PERFORMANCE_MODEL__TYPES_H_

namespace performance_model{

// We adopt Steven's typology for parameter types. Categorical comprises ordinal and nominal type. As interpolation and generalization across distinct values is not relevant for such parameters, we don't differentiate between the two. Numerical comprises ratio and interval, as both interpolation and extrapolation are relevant for both parameter types.
enum class parameter_type { CATEGORICAL, NUMERICAL };

enum class runtime_transformation { NONE, LOG };
enum class parameter_transformation { NONE, LOG };
enum class loss_function { MSE, MLOGQ2 };

};
#endif // __PERFORMANCE_MODEL__TYPES_H_
