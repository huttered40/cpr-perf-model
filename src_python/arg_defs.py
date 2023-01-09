def add_general_arguments(parser):

    parser.add_argument(
        '--write_header',
        type=int,
        default='0',
        metavar='int',
        help='Boolean decision whether to write column headers to CSV file (default: 0)')
    parser.add_argument(
        '--scale_mode',
        type=str,
        default='strong',
        metavar='str',
        help='Kernel name (default: )')
    parser.add_argument(
        '--training_set_size',
        type=int,
        default=0,
        metavar='int',
        help='Number of randomly sampled inputs constrained to grid-points (default: 0). Typically a smaller integer than training_set_id')
    parser.add_argument(
        '--test_set_size',
        type=int,
        default=0,
        metavar='int',
        help='Number of randomly sampled inputs (default: 0)')
    parser.add_argument(
        '--ngrid_pts',
        type=str,
        default='',
        metavar='str',
        help='ID for discretization granularity of kernel configuration space. It can have a different meaning across kernels (default: 0)')
    parser.add_argument(
        '--check_training_error',
        type=int,
        default="0",
        metavar='int',
        help='')
    parser.add_argument(
        '--cell_spacing',
        type=str,
        default="0",
        metavar='str',
        help='ID for placement of grid-points constrained to a particular discretization granularity as specified by grid_type. Equivalently, ID for sampling distribution (default: 0)')
    parser.add_argument(
        '--training_set_split_percentage',
        type=float,
        default='0',
        metavar='float',
        help='Percentage of the training set used for model selection across hyper-parameter space (default: 0)')
    parser.add_argument(
        '--response_transform',
        type=int,
        default="1",
        metavar='int',
        help='Transformation to apply to runtime data (default: 1 (Log transformation))')
    parser.add_argument(
        '--max_spline_degree',
        type=int,
        default="1",
        metavar='int',
        help='Maximum spline degree for extrapolation model (default: 1)')
    parser.add_argument(
        '--build_extrapolation_model',
        type=int,
        default="1",
        metavar='int',
        help='Signifies whether to build a separate model for extrapolation (default: 1)')
    parser.add_argument(
        '--reg',
        type=str,
        default='1e-5',
        metavar='str',
        help='regularization parameter (default: 1e-6)')
    parser.add_argument(
        '--max_num_sweeps',
        type=int,
        default='20',
        metavar='str',
        help='Maximum number of sweeps of ALS or AMN (default: 20)')
    parser.add_argument(
        '--sweep_tol',
        type=float,
        default='1e-3',
        metavar='float',
        help='Tolerance for ALS or AMN (default: 1e-3)')
    parser.add_argument(
        '--barrier_start',
        type=float,
        default='100',
        metavar='float',
        help='Coefficient on barrier terms for initial ALS sweep (default: 1000)')
    parser.add_argument(
        '--barrier_reduction_factor',
        type=float,
        default='1.25',
        metavar='float',
        help='Divisor for coefficient on barrier terms for subsequent ALS sweeps (default: 10)')
    parser.add_argument(
        '--tol_newton',
        type=float,
        default='1e-3',
        metavar='float',
        help='Tolerance for Newtons method (default: 1e-3)')
    parser.add_argument(
        '--max_num_newton_iter',
        type=int,
        default='40',
        metavar='float',
        help='Max number of iterations of Newtons method (default: 20)')
    parser.add_argument(
        '--cp_rank',
        type=str,
        default="3",
        metavar='str',
        help='Comma-delimited list of CP ranks (default: 1,2,3)')
    parser.add_argument(
        '--numpy_eval',
        type=int,
        default=1,
        metavar='int',
        help='ID specifying whether to use Numpy for evaluation. Only valid when using a single process and will switch automatically (default: 1)')
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
    parser.add_argument(
        '--print_diagnostics',
        type=int,
        default=0,
        metavar='int',
        help='Whether or not to print default and input information (default: 0)')
