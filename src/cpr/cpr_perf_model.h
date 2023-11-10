#include <string>
#include <vector>
#include <array>

namespace performance_model{

// cpr_perf_model: Canonical-Polyadic decomposition of Regular grids storing runtimes as a PERFormance MODEL
class cpr_perf_model : public model{
  public:
    perf_model(int order, std::vector<parameter_type> param_types,
      std::vector<parameter_range_partition> interval_spacing);
    perf_model(std::string& file_path);
/*
    cp_perf_model(int order, const std::vector<int>& cells_info, std::vector<parameter_type> param_types,
      std::vector<parameter_range_partition> interval_spacing, const std::vector<double>& mode_range_min,
      const std::vector<double>& mode_range_max, std::vector<int> cp_rank={3,3},
      int response_transform=1, double max_spacing_factor=2.0, std::vector<double> custom_grid_pts={},
      bool save_dataset=false);
*/
    ~perf_model();

    double predict(const double* configuration, const double* _factor_matrix_elements_generalizer_=nullptr, const double* _factor_matrix_elements_extrapolator_=nullptr);
    bool train(int num_configurations, const double* configurations, const double* runtimes, cp_param_pack* arg_pack=nullptr, bool compute_fit_error=true, bool save_dataset=false);

    double timer1,timer2,timer3,timer4,timer5,timer6,timer7;

  protected:
    void init(std::vector<int>& cells_info, const std::vector<double>& mode_range_min,
      const std::vector<double>& mode_range_max, double max_spacing_factor, const std::vector<double> custom_grid_pts={},
      int num_configurations=-1, const double* features=nullptr);

    bool m_is_valid;
    int order;
    int max_spline_degree;
    runtime_transformation response_transform;
    parameter_transformation feature_transform;
    std::array<int,2> cp_rank;
    std::vector<int> tensor_mode_lengths;
    std::vector<parameter_range_partition> interval_spacing;
    std::vector<parameter_range_partition> fixed_interval_spacing;
    std::vector<parameter_type> param_types;
    std::vector<double> tensor_cell_nodes;
    std::vector<int> tensor_cell_node_offsets;
    std::vector<int> numerical_modes;
    std::vector<int> categorical_modes;
    double* factor_matrix_elements_generalizer;
    double* factor_matrix_elements_extrapolator;
    double* factor_matrix_elements_extrapolator_models;
    //NOTE: Below: not sure whether to store all configurations
    std::vector<double> yi;
    std::vector<double> Xi;
    std::vector<std::vector<int>> Projected_Omegas;
};
};
