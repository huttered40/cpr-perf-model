export CPPM_VERBOSE=1

# Interval spacing (and corresponding interpolation and quadrature methods) for parameters of type NUMERICAL
# Options: SINGLE,AUTOMATIC,GEOMETRIC,UNIFORM
export CPPMI_PARTITION_SPACING=GEOMETRIC
export CPPME_PARTITION_SPACING=GEOMETRIC

# Number of partitions of a numerical parameter's range.
export CPPMI_PARTITIONS_PER_DIMENSION=32
export CPPME_PARTITIONS_PER_DIMENSION=32

# Number of observations to inclde per partition of a numerical parameter's range when considering AUTOMATIC spacing for NUMERICAL parameters
# The smaller we set this, the smaller quadrature error should be and the more irregular the spacing between nodes and larger the tensor. Yet, training the model will be more difficult.
export CPPMI_OBS_PER_PARTITION=512
export CPPME_OBS_PER_PARTITION=512

# CP rank of model trained with loss function CPPM_LOSS_FUNCTION
export CPPMI_CP_RANK=12
export CPPME_CP_RANK=2

# Type of transformation (NONE,LOG) to apply to observed tensor elements before training model with loss function CPPM_LOSS_FUNCTION
export CPPMI_RUNTIME_TRANSFORM=LOG
export CPPME_RUNTIME_TRANSFORM=NONE

# Max ratio of cell size (end_pos/start_pos) to consider for NUMERICAL parameters with AUTOMATIC spacing
# The smaller we set this, the smaller the quadrature error should be, especially if each parameter/mode represents estimated costs.
# Note that even if this factor is 2, the quadrature is HIGH, yet if small-enough (like 1.25 or 1.1), the quadrature error is small! This is good! No the onus is on getting the loss value low as well.
export CPPMI_MAX_SPACING_FACTOR=2
export CPPME_MAX_SPACING_FACTOR=2

# Choice of loss function for optimizing CPD model (used in interpolation settings). Choices: MSE or MLogQ2
export CPPMI_LOSS_FUNCTION=MSE
export CPPME_LOSS_FUNCTION=MLogQ2

# Regularization coefficient used in training model with loss function MSE
export CPPMI_REGULARIZATION=1e-8
export CPPME_REGULARIZATION=1e-4

export CPPMI_OPTIMIZATION_BARRIER_START=1e-1
export CPPME_OPTIMIZATION_BARRIER_START=1e-1

export CPPMI_OPTIMIZATION_BARRIER_STOP=1e-11
export CPPME_OPTIMIZATION_BARRIER_STOP=1e-11

export CPPMI_OPTIMIZATION_BARRIER_REDUCTION_FACTOR=100
export CPPME_OPTIMIZATION_BARRIER_REDUCTION_FACTOR=100

export CPPMI_FM_CONVERGENCE_TOL=1e-3
export CPPME_FM_CONVERGENCE_TOL=1e-3

export CPPMI_FM_MAX_NUM_ITER=10
export CPPME_FM_MAX_NUM_ITER=10

export CPPMI_MAX_NUM_SWEEPS=4
export CPPME_MAX_NUM_SWEEPS=4

export CPPMI_SWEEP_TOL=1e-2
export CPPME_SWEEP_TOL=1e-2

export CPPMI_MAX_NUM_RE_INITS=1
export CPPME_MAX_NUM_RE_INITS=1

# This differs from MIN_NUM_OBS_PER_MODEL_UPDATE because it is internal to cp-perf-model, and is checked even if CPPMI_AGGREGATE_OBS_ACROSS_COMM is set
export CPPMI_MIN_NUM_OBS_FOR_TRAINING=64
export CPPME_MIN_NUM_OBS_FOR_TRAINING=64

export CPPMI_OPTIMIZATION_CONVERGENCE_TOLERANCE_FOR_RE_INIT=1e-1
export CPPME_OPTIMIZATION_CONVERGENCE_TOLERANCE_FOR_RE_INIT=1e-1

export CPPMI_INTERPOLATION_FACTOR_TOL=0.5
export CPPME_INTERPOLATION_FACTOR_TOL=0.5

# This option enables aggregation of all samples (AGGREGATE_REPEAT_OBS should be set to avoid excess communication cost), which each process can use for distributed training
# Also note that if this is set, we do not check the sample count in CTF (src/shared/model.cxx) before we enter cp-perf-model::train(...).
export CPPMI_AGGREGATE_OBS_ACROSS_COMM=0
export CPPME_AGGREGATE_OBS_ACROSS_COMM=0

# The maximum degree of spline model of left singular vector of rank-1 SVD of each factor matrix comprising CP decomposition corresponding to a numerical model parameter.
export CPPME_MAX_SPLINE_DEGREE=2

# Choice between global univariate model or spline univariate model of left singular vector of rank-1 SVD of each factor matrix comprising CP decomposition corresponding to a numerical model parameter.
export CPPME_USE_SPLINE=0


