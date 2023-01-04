def add_general_arguments(parser):

    parser.add_argument(
        '--write_header',
        type=int,
        default='0',
        metavar='int',
        help='Boolean decision whether to write column headers to CSV file (default: 0)')
    parser.add_argument(
        '--dense_grid_exist',
        type=int,
        default='0',
        metavar='int',
        help='Boolean decision whether to test completion against dense grid (default: 0)')
    parser.add_argument(
        '--grid_test_set',
        type=int,
        default='0',
        metavar='int',
        help='')
    parser.add_argument(
        '--kernel_name',
        type=str,
        default='poly',
        metavar='str',
        help='Kernel name (default: )')
    parser.add_argument(
        '--gp_kernel_id',
        type=str,
        default='2',
        metavar='str',
        help='Gaussian Process kernel (see gp.py) (default: 2)')
    parser.add_argument(
        '--scale_mode',
        type=str,
        default='strong',
        metavar='str',
        help='Kernel name (default: )')
    parser.add_argument(
        '--transfer_learning_mode',
        type=str,
        default='',
        metavar='',
        help=' (default: )')
    parser.add_argument(
        '--kernel_type',
        type=int,
        default=0,
        metavar='int',
        help='ID specifying task, hardware, and/or tuning parameters to evaluate (default: 0)')
    parser.add_argument(
        '--noise_level',
        type=float,
        default=0,
        metavar='float',
        help='ID specifying task, hardware, and/or tuning parameters to evaluate (default: 0)')
    parser.add_argument(
        '--thread_count',
        type=int,
        default=0,
        metavar='int',
        help='ID specifying task, hardware, and/or tuning parameters to evaluate (default: 0)')
    parser.add_argument(
        '--ppn',
        type=int,
        default=0,
        metavar='int',
        help='ID specifying task, hardware, and/or tuning parameters to evaluate (default: 0)')
    parser.add_argument(
        '--node_count',
        type=int,
        default=0,
        metavar='int',
        help='ID specifying task, hardware, and/or tuning parameters to evaluate (default: 0)')
    parser.add_argument(
        '--training_set_niter',
        type=int,
        default=0,
        metavar='int',
        help='Number of executions per input tuple for training set (default: 0)')
    parser.add_argument(
        '--test_set_niter',
        type=int,
        default=0,
        metavar='int',
        help='Number of executions per input tuple for test set (default: 0)')
    parser.add_argument(
        '--training_set_size',
        type=int,
        default=0,
        metavar='int',
        help='Number of randomly sampled inputs constrained to grid-points (default: 0). Typically a smaller integer than training_set_id')
    parser.add_argument(
        '--training_set_id',
        type=int,
        default=0,
        metavar='int',
        help='Training set File ID (default: 0)')
    parser.add_argument(
        '--test_set_size',
        type=int,
        default=0,
        metavar='int',
        help='Number of randomly sampled inputs (default: 0)')
    parser.add_argument(
        '--grid_type',
        type=int,
        default="0",
        metavar='int',
        help='ID for discretization granularity of kernel configuration space. It can have a different meaning across kernels (default: 0)')
    parser.add_argument(
        '--sample_type',
        type=int,
        default="0",
        metavar='int',
        help='ID for discretization granularity of kernel configuration space. It can have a different meaning across kernels (default: 0)')
    parser.add_argument(
        '--cell_counts',
        type=str,
        default='',
        metavar='str',
        help='ID for discretization granularity of kernel configuration space. It can have a different meaning across kernels (default: 0)')
    parser.add_argument(
        '--cell_size',
        type=int,
        default="1",
        metavar='int',
        help='')
    parser.add_argument(
        '--check_training_error',
        type=int,
        default="0",
        metavar='int',
        help='')
    parser.add_argument(
        '--grid_types',
        type=str,
        default="0",
        metavar='str',
        help='ID for discretization granularity of kernel configuration space. It can have a different meaning across kernels (default: 0)')
    parser.add_argument(
        '--cell_spacing',
        type=str,
        default="0",
        metavar='str',
        help='ID for placement of grid-points constrained to a particular discretization granularity as specified by grid_type. Equivalently, ID for sampling distribution (default: 0)')
    parser.add_argument(
        '--test_set_split_percentage',
        type=float,
        default='.1',
        metavar='float',
        help='Percentage of the test set used for model validation across hyper-parameter space (default: 0.1)')
    parser.add_argument(
        '--response_transform',
        type=int,
        default="1",
        metavar='int',
        help='Transformation to apply to runtime data (default: 1 (Log transformation))')
    parser.add_argument(
        '--error_metric',
        type=str,
        default='MSE',
        metavar='str',
        help='Error metric characterizing loss function (default: MSE)')
    parser.add_argument(
        '--predictor_transform',
        type=int,
        default='0',
        metavar='int',
        help='Transformation to apply to predictors (default: 0 (No transformation))')
    parser.add_argument(
        '--reg',
        type=str,
        default='1e-6',
        metavar='str',
        help='regularization parameter (default: 1e-6)')
    parser.add_argument(
        '--tol_als',
        type=float,
        default='1e-3',
        metavar='float',
        help='Tolerance for ALS (default: 1e-3)')
    parser.add_argument(
        '--nals_sweeps',
        type=str,
        default='20',
        metavar='str',
        help='Maximum number of sweeps of ALS (default: 20)')
    parser.add_argument(
        '--barrier_start',
        type=float,
        default='1000',
        metavar='float',
        help='Coefficient on barrier terms for initial ALS sweep (default: 1000)')
    parser.add_argument(
        '--barrier_reduction_factor',
        type=float,
        default='10',
        metavar='float',
        help='Divisor for coefficient on barrier terms for subsequent ALS sweeps (default: 10)')
    parser.add_argument(
        '--tol_newton',
        type=float,
        default='1e-3',
        metavar='float',
        help='Tolerance for Newtons method (default: 1e-3)')
    parser.add_argument(
        '--max_iter_newton',
        type=int,
        default='20',
        metavar='float',
        help='Max number of iterations of Newtons method (default: 20)')
    parser.add_argument(
        '--cp_rank',
        type=str,
        default="1,2,3",
        metavar='str',
        help='Comma-delimited list of CP ranks (default: 1,2,3)')
    parser.add_argument(
        '--element_mode_len',
        type=str,
        default='2',
        metavar='str',
        help='Comma-delimited tuple representing the number of grid-points along each tensor mode to use for interpolation (default: 2)')
    parser.add_argument(
        '--numpy_eval',
        type=int,
        default=1,
        metavar='int',
        help='ID specifying whether to use Numpy for evaluation. Only valid when using a single process and will switch automatically (default: 1)')
    parser.add_argument(
        '--tensor_dim',
        type=int,
        default=0,
        metavar='int',
        help='Number of tensor modes for CP-ALS (can be less than or equal to number of kernel parameters (default: 0)')
    parser.add_argument(
        '--tensor_map',
        type=str,
        default='',
        metavar='str',
        help='Comma-delimited tuple representing the mapping of parameters to tensor modes (default: )')
    parser.add_argument(
        '--interp_map',
        type=str,
        default='',
        metavar='str',
        help='Comma-delimited tuple representing which parameter modes to interpolate (default: )')

    parser.add_argument(
        '--nbases',
        type=str,
        default='1',
        metavar='str',
        help='Comma-delimited tuple representing the number of bases in a least-squares method (what each basis refers to is kernel-specific; see files) (default: 1)')
    parser.add_argument(
        '--spline_degrees',
        type=str,
        default="1,2,3,4,5",
        metavar='str',
        help='Comma-delimited list of number of degrees for splines to include for a MARS model (default: 1,2,3,4,5)')
    parser.add_argument(
        '--tree_depth',
        type=str,
        default="10",
        metavar='str',
        help='Comma-delimited list (default: 1,2,3,4,5)')
    parser.add_argument(
        '--ntrees',
        type=str,
        default="100",
        metavar='str',
        help='Comma-delimited list (default: 1,2,3,4,5)')
    parser.add_argument(
        '--nneighbors',
        type=str,
        default="5",
        metavar='str',
        help='Comma-delimited list (default: 1,2,3,4,5)')
    parser.add_argument(
        '--nlevels',
        type=str,
        default='3',
        metavar='str',
        help='Comma-delimited list of number of sparse grid levels (default: 3)')
    parser.add_argument(
        '--nadaptpts',
        type=str,
        default='3',
        metavar='str',
        help='Comma-delimited list of number of grid-points to update (default: 3)')
    parser.add_argument(
        '--nrefinements',
        type=int,
        default='5',
        metavar='int',
        help='Number of sparse-grid refinements (default: 5)')
    parser.add_argument(
        '--test_per_refinement',
        type=int,
        default='1',
        metavar='int',
        help='Signifies whether to test model on validation set with each refinement of grid before further refinement (default: 1)')


    parser.add_argument(
        '--make_heat_plot',
        type=int,
        default="0",
        metavar='int',
        help='')
    parser.add_argument(
        '--analyze_hyperplanes',
        type=int,
        default=0,
        metavar='int',
        help='Number of randomly sampled inputs')
    parser.add_argument(
        '--analyze_projections',
        type=int,
        default=0,
        metavar='int',
        help='Number of randomly sampled inputs')
    parser.add_argument(
        '--error_mode',
        type=int,
        default=0,
        metavar='int',
        help='Number of randomly sampled inputs')
    parser.add_argument(
        '--fold_id',
        type=int,
        default=0,
        metavar='int',
        help='')
    parser.add_argument(
        '--hidden_layer_sizes',
        type=str,
        default='100',
        metavar='str',
        help='Comma-delimited list of number of grid-points to update (default: 3)')
    parser.add_argument(
        '--activation',
        type=str,
        default='relu',
        metavar='str',
        help='Comma-delimited list of number of grid-points to update (default: 3)')
    parser.add_argument(
        '--solver',
        type=str,
        default='adam',
        metavar='str',
        help='Comma-delimited list of number of grid-points to update (default: 3)')
    parser.add_argument(
        '--training_file',
        type=str,
        default='',
        metavar='str',
        help='Full file path to training data')
    parser.add_argument(
        '--test_file',
        type=str,
        default='',
        metavar='str',
        help='Full file path to test data')
    parser.add_argument(
        '--output_file',
        type=str,
        default='',
        metavar='str',
        help='Full file path to output data')
    parser.add_argument(
        '--input_columns',
        type=str,
        default='',
        metavar='str',
        help='Comma-delimited list of column indices corresponding to training/test inputs')
    parser.add_argument(
        '--data_columns',
        type=str,
        default='',
        metavar='str',
        help='Comma-delimited list of column indices corresponding to training/test data')
    parser.add_argument(
        '--mode_range_min',
        type=str,
        default='',
        metavar='str',
        help='Comma-delimited list of minimum values for each parameter')
    parser.add_argument(
        '--mode_range_max',
        type=str,
        default='',
        metavar='str',
        help='Comma-delimited list of maximum values for each parameter')
    parser.add_argument(
        '--print_model_parameters',
        type=int,
        default=0,
        metavar='int',
        help='Whether or not to print the factor matrix elements (default: 0)')
