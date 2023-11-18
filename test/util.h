#include <string>

#include "cp_perf_model.h"

/*
#include <format>
#include <string_view>

template<typename... Args>
void print(const std::string_view fmt_str, Args&&... args){
  //auto fmt_args{std::make_format_args(args...)};// Interesting that we do not forward
  auto fmt_args{std::make_format_args(std::forward<Args>(args)...)};
  std::string outstr{ vformat(fmt_str,fmt_args) };
  fputs(outstr.c_str(),stdout);
}
*/

struct evaluation_info{
  double avg_inference_latency{0};
  double max_inference_latency{0};
  // mlogq: arithmetic mean of log-ratios
  double mlogq_error{0};
  double max_logq_error{0};
  // mlogqabs: arithmetic mean of absolute values of log-ratios
  double mlogqabs_error{0};
  double max_logqabs_error{0};
  // mlogq2: arithmetic mean of the square of log-ratios
  double mlogq2_error{0};
  double max_logq2_error{0};
  // maps: arithmetic mean of the absolute percentage error
  double maps_error{0};
  double max_aps_error{0};
};

void shuffle_runtimes(int nc, int nparam, std::vector<double>& runtimes, std::vector<double>& configurations);

template<typename T>
void print(const char* msg, T val){
  std::cout << msg << val << "\n";
}

template<typename T, typename U>
void print(const char* msg, T val1, U val2){
  std::cout << msg << val1 << ": " << val2 << "\n";
}

void print_model_info(const performance_model::cpr_model_fit_info& info);

void custom_assert(bool alert, const char* msg);

double get_wall_time();

std::vector<std::string> get_cpr_model_hyperparameter_options();

void set_cpr_param_pack(int nparam, performance_model::cpr_hyperparameter_pack& arg_pack, std::vector<std::string>&& hyperparameter_options = get_cpr_model_hyperparameter_options(), bool verbose=false);

void set_cprg_param_pack(int nparam, performance_model::cprg_hyperparameter_pack& arg_pack, bool verbose=false);

bool is_verbose();

void evaluate(int nparam, int size, std::vector<double>& runtimes, std::vector<double>& configurations, performance_model::model* interpolator, performance_model::model* extrapolator,
              const performance_model::cpr_hyperparameter_pack& interpolator_pack,
              const performance_model::cprg_hyperparameter_pack& extrapolator_pack,
              const performance_model::cpr_model_fit_info& interpolator_info,
              const performance_model::cpr_model_fit_info& extrapolator_info,
              const char* file_name,
              bool verbose=false);
