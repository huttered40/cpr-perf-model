#ifndef __PERFORMANCE_MODEL__CPR_TYPES_H_
#define __PERFORMANCE_MODEL__CPR_TYPES_H_

namespace performance_model{

// Note that categorical parameters use uniform or custom spacing by default.
enum class parameter_range_partition { UNIFORM, GEOMETRIC, CUSTOM, AUTOMATIC, SINGLE };

};

#endif // __PERFORMANCE_MODEL__CPR_TYPES_H_
