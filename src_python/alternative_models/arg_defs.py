def add_general_arguments(parser):

    parser.add_argument(
        '--write_header',
        type=int,
        default='0',
        metavar='int',
        help='Boolean decision whether to write column headers to CSV file (default: 0)')
    parser.add_argument(
        '--gp_kernel_id',
        type=str,
        default='2',
        metavar='str',
        help='Gaussian Process kernel (see gp.py) (default: 2)')
    parser.add_argument(
        '--kernel',
        type=str,
        default='poly',
        metavar='str',
        help='SVM kernel (see svm.py) (default: poly)')
    parser.add_argument(
        '--kernel_type',
        type=int,
        default=0,
        metavar='int',
        help='ID specifying task, hardware, and/or tuning parameters to evaluate (default: 0)')
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
        '--training_set_split_percentage',
        type=float,
        default='.0',
        metavar='float',
        help='Percentage of the training set used for model selection across hyper-parameter space (default: 0.0)')
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
        default="16",
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
    parser.add_argument(
        '--print_diagnostics',
        type=int,
        default=0,
        metavar='int',
        help='Whether or not to print default and input information (default: 0)')
