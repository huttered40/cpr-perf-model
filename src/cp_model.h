//TODO: Sequential programs won't use MPI
#ifdef CRITTER
#include "critter_mpi.h"
#include "critter_symbol.h"
#else
#include <mpi.h>
#endif

#include <string>
#include <fstream>
#include <vector>
#include <array>
#include <cassert>

// We adopt Steven's typology for parameter types. Categorical comprises ordinal and nominal type. As interpolation and generalization across distinct values is not relevant for such parameters, we don't differentiate between the two. Numerical comprises ratio and interval, as both interpolation and extrapolation are relevant for both parameter types.
enum MODEL_PARAM_TYPE : int { CATEGORICAL, NUMERICAL };
// Note that categorical parameters use uniform or custom spacing by default.
enum MODEL_PARAM_SPACING : int { UNIFORM, GEOMETRIC, CUSTOM, AUTOMATIC, SINGLE };

class cp_perf_model{
  public:
    cp_perf_model(int order, std::vector<MODEL_PARAM_TYPE> param_types,
      std::vector<MODEL_PARAM_SPACING> interval_spacing, std::vector<int> cp_rank={3,3}, int response_transform=1, bool save_dataset=false);
    cp_perf_model(std::string& file_path);
    cp_perf_model(int order, const std::vector<int>& cells_info, std::vector<MODEL_PARAM_TYPE> param_types,
      std::vector<MODEL_PARAM_SPACING> interval_spacing, const std::vector<double>& mode_range_min,
      const std::vector<double>& mode_range_max, std::vector<int> cp_rank={3,3},
      int response_transform=1, double max_spacing_factor=2.0, std::vector<double> custom_grid_pts={},
      bool save_dataset=false);
    ~cp_perf_model();

    double predict(const double* configuration, const double* _factor_matrix_elements_generalizer_=nullptr, const double* _factor_matrix_elements_extrapolator_=nullptr);
    bool train(bool compute_fit_error, double* loss, int num_configurations, const double* configurations, const double* runtimes, MPI_Comm cm_training, MPI_Comm cm_data, std::vector<int>& cells_info, const char* loss_function="MSE", bool aggregate_obs_across_comm=false, double max_spacing_factor=2.0, std::vector<double> regularization={1e-4,1e-4}, int max_spline_degree=3, std::vector<double> model_convergence_tolerance={1e-3,1e-2},
      std::vector<int> maximum_num_sweeps={100,20}, double factor_matrix_convergence_tolerance=1e-3,
      int maximum_num_iterations=40, std::vector<double> barrier_range={1e-11,1e1},
      double barrier_reduction_factor=8, std::vector<int> projection_set_size_threshold_={});

    int get_cp_rank();

    double timer1,timer2,timer3,timer4,timer5,timer6,timer7;

  private:
    void init(std::vector<int>& cells_info, const std::vector<double>& mode_range_min,
      const std::vector<double>& mode_range_max, double max_spacing_factor, const std::vector<double> custom_grid_pts={},
      int num_configurations=-1, const double* features=nullptr);

    bool is_valid;
    int max_spline_degree;
    bool save_dataset;
    int order;
    int response_transform;
    std::array<int,2> cp_rank;
    std::vector<int> tensor_mode_lengths;
    std::vector<MODEL_PARAM_SPACING> interval_spacing;
    std::vector<MODEL_PARAM_SPACING> fixed_interval_spacing;
    std::vector<MODEL_PARAM_TYPE> param_types;
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
