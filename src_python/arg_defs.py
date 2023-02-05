def add_general_arguments(parser):

    parser.add_argument(
        '--write_header',
        type=int,
        default='0',
        metavar='int',
        help='Boolean decision whether to write column headers to CSV file (default: 0)')
    parser.add_argument(
        '--training_set_size',
        type=int,
        default=0,
        metavar='int',
        help='Size of training set (default: 0).')
    parser.add_argument(
        '--test_set_size',
        type=int,
        default=0,
        metavar='int',
        help='Size of test set (default: 0).')
    parser.add_argument(
        '--ngrid_pts',
        type=str,
        default='2',
        metavar='str',
        help='ID for discretization granularity of kernel configuration space (default: 2).')
    parser.add_argument(
        '--cell_spacing',
        type=str,
        default="1",
        metavar='str',
        help='ID for placement of grid-points constrained to a particular discretization granularity (not necessarily equivalent to sampling distribution of training dataset (default: 1).')
    parser.add_argument(
        '--training_set_split_percentage',
        type=float,
        default='0',
        metavar='float',
        help='Percentage of the training set used for model selection across hyper-parameter space (default: 0).')
    parser.add_argument(
        '--response_transform',
        type=int,
        default="1",
        metavar='int',
        help='Transformation to apply to runtime data (default: 1 (Log transformation)).')
    parser.add_argument(
        '--max_spline_degree',
        type=int,
        default="1",
        metavar='int',
        help='Maximum spline degree for extrapolation model (default: 1).')
    parser.add_argument(
        '--build_extrapolation_model',
        type=int,
        default="1",
        metavar='int',
        help='Signifies whether to build a separate model for extrapolation (default: 1).')
    parser.add_argument(
        '--reg',
        type=str,
        default='1e-4',
        metavar='str',
        help='regularization coefficient (default: 1e-5).')
    parser.add_argument(
        '--max_num_sweeps',
        type=int,
        default='100',
        metavar='str',
        help='Maximum number of sweeps of alternating minimization (default: 20).')
    parser.add_argument(
        '--sweep_tol',
        type=float,
        default='1e-5',
        metavar='float',
        help='Error tolerance for alternating minimization method (default: 1e-3).')
    parser.add_argument(
        '--barrier_start',
        type=float,
        default='1e1',
        metavar='float',
        help='Coefficient on barrier terms for initial sweep of Alternating Minimization via Newtons method (default: 100).')
    parser.add_argument(
        '--barrier_stop',
        type=float,
        default='1e-9',
        metavar='float',
        help='Coefficient on barrier terms for initial sweep of Alternating Minimization via Newtons method (default: 100).')
    parser.add_argument(
        '--barrier_reduction_factor',
        type=float,
        default='2',
        metavar='float',
        help='Divisor for coefficient on barrier terms for subsequent sweeps of Alternating Minimization via Newtons method (default: 1.25).')
    parser.add_argument(
        '--tol_newton',
        type=float,
        default='1e-3',
        metavar='float',
        help='Change (in factor matrix) tolerance within Newtons method (default: 1e-3).')
    parser.add_argument(
        '--max_num_newton_iter',
        type=int,
        default='40',
        metavar='float',
        help='Maximum number of iterations of Newtons method (default: 40)')
    parser.add_argument(
        '--cp_rank',
        type=str,
        default="3",
        metavar='str',
        help='Comma-delimited list of Canonical-Polyadic tensor decomposition ranks (default: 3).')
    parser.add_argument(
        '--cp_rank_for_extrapolation',
        type=int,
        default="1",
        metavar='int',
        help='Canonical-Polyadic tensor decomposition rank for use in extrapolation (default: 1).')
    parser.add_argument(
        '--loss_function',
        type=int,
        default="0",
        metavar='int',
        help='Loss function to optimize CPD Model for interpolation environment.')
    parser.add_argument(
        '--interp_map',
        type=str,
        default='',
        metavar='str',
        help='Comma-delimited list signifying which parameter ranges to interpolate (default: ).')
    parser.add_argument(
        '--training_file',
        type=str,
        default='',
        metavar='str',
        help='File path to training dataset.')
    parser.add_argument(
        '--test_file',
        type=str,
        default='',
        metavar='str',
        help='File path to test dataset.')
    parser.add_argument(
        '--output_file',
        type=str,
        default='',
        metavar='str',
        help='File path to write prediction results.')
    parser.add_argument(
        '--input_columns',
        type=str,
        default='',
        metavar='str',
        help='Comma-delimited list of column indices corresponding to each parameter witin a configuration for both training and test datasets.')
    parser.add_argument(
        '--data_columns',
        type=str,
        default='',
        metavar='str',
        help='Comma-delimited list of column indices corresponding to response (execution time) within training and test datasets (same format assumed for both).')
    parser.add_argument(
        '--mode_range_min',
        type=str,
        default='',
        metavar='str',
        help='Comma-delimited list of minimum values for each parameter within a configuration.')
    parser.add_argument(
        '--mode_range_max',
        type=str,
        default='',
        metavar='str',
        help='Comma-delimited list of maximum values for each parameter within a configuration.')
    parser.add_argument(
        '--print_model_parameters',
        type=int,
        default=0,
        metavar='int',
        help='Whether or not to print the factor matrix elements (default: 0).')
    parser.add_argument(
        '--print_diagnostics',
        type=int,
        default=0,
        metavar='int',
        help='Whether or not to print information about datasets (default: 0).')
    parser.add_argument(
        '--print_test_error',
        type=int,
        default="0",
        metavar='int',
        help='')
