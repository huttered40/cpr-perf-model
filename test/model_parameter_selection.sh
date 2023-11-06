# Interval spacing (and corresponding interpolation and quadrature methods) for parameters of type NUMERICAL
export CPPM_INTERVAL_SPACING=GEOMETRIC

# Number of partitions of a numerical parameter's range.
export CPPM_CELLS_PER_DIMENSION=32

# Number of observations to inclde per partition of a numerical parameter's range when considering AUTOMATIC spacing for NUMERICAL parameters
# The smaller we set this, the smaller quadrature error should be and the more irregular the spacing between nodes and larger the tensor. Yet, training the model will be more difficult.
export CPPM_OBS_PER_CELL=32

# CP rank of model trained with loss function CPPM_LOSS_FUNCTION
export CPPM_CP_RANK_1=12

# CP rank of model trained with loss function MLogQ2
export CPPM_CP_RANK_2=2

# Type of transformation (1: log, 0: None) to apply to observed tensor elements before training model with loss function CPPM_LOSS_FUNCTION
export CPPM_RESPONSE_TRANSFORM_ID=1

# Max ratio of cell size (end_pos/start_pos) to consider for NUMERICAL parameters with AUTOMATIC spacing
# The smaller we set this, the smaller the quadrature error should be, especially if each parameter/mode represents estimated costs.
# Note that even if this factor is 2, the quadrature is HIGH, yet if small-enough (like 1.25 or 1.1), the quadrature error is small! This is good! No the onus is on getting the loss value low as well.
export CPPM_MAX_SPACING_FACTOR=1.25

# Choice of loss function for optimizing CPD model (used in interpolation settings). Choices: MSE or MLogQ2
export CPPM_LOSS_FUNCTION=MSE

# Regularization coefficient used in training model with loss function MSE
export CPPM_REG_1=0

# Regularization coefficient used in training model with loss function MLogQ2
export CPPM_REG_2=1e-3

# The maximum degree of spline model of left singular vector of rank-1 SVD of each factor matrix comprising CP decomposition corresponding to a numerical model parameter.
export CPPM_MAX_SPLINE_DEGREE=1

export CPPM_BARRIER_START=1e-1
export CPPM_BARRIER_STOP=1e-11
export CPPM_BARRIER_REDUCTION_FACTOR=1e2
export CPPM_FM_CONVERGENCE_TOL=1e-3
export CPPM_FM_MAX_NUM_ITER=10
export CPPM_MAX_NUM_SWEEPS_1=10
export CPPM_MAX_NUM_SWEEPS_2=2
export CPPM_SWEEP_TOL_1=1e-2
export CPPM_SWEEP_TOL_2=1e-2
export CPPM_VERBOSE=1

# These four environment variables specified below are queried within the source code of cp_model.cxx
export CPPM_MAX_NUM_RE_INITS=1
export CPPM_RE_INIT_LOSS_TOL=.1
export CPPM_INTERPOLATION_FACTOR_TOL=0.5
# This differs from MIN_NUM_OBS_PER_MODEL_UPDATE because it is internal to cp-perf-model, and is checked even if CPPM_AGGREGATE_OBS_ACROSS_COMM is set
export CPPM_MIN_NUM_OBS_FOR_TRAINING=64
# Choice between global univariate model or spline univariate model of left singular vector of rank-1 SVD of each factor matrix comprising CP decomposition corresponding to a numerical model parameter.
export CPPM_USE_SPLINE=0


# Below: Not Relevant

# This option enables aggregation of all samples (AGGREGATE_REPEAT_OBS should be set to avoid excess communication cost), which each process can use for distributed training
# Also note that if this is set, we do not check the sample count in CTF (src/shared/model.cxx) before we enter cp-perf-model::train(...).
export CPPM_AGGREGATE_OBS_ACROSS_COMM=0

# Below: if ID is 0, then use CELLS_PER_DIMENSION for each numerical mode. if ID is 1, use OBS_PER_CELL for each numerical mode
export CPPM_TENSOR_MODE_SIZE_ID=1

