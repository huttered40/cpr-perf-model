def add_general_arguments(parser):

    parser.add_argument(
        '--write_header',
        type=int,
        default='0',
        metavar='int',
        help='Boolean decision whether to write column headers to CSV file (default: 0).')
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
        '--gp_kernel_id',
        type=str,
        default='2',
        metavar='str',
        help='Gaussian Process kernel (see gp.py) (default: 2).')
    parser.add_argument(
        '--kernel',
        type=str,
        default='poly',
        metavar='str',
        help='SVM kernel (see svm.py) (default: poly).')
    parser.add_argument(
        '--kernel_type',
        type=int,
        default=0,
        metavar='int',
        help='ID specifying task, hardware, and/or tuning parameters to evaluate (default: 0).')
    parser.add_argument(
        '--error_metric',
        type=str,
        default='MSE',
        metavar='str',
        help='Error metric characterizing loss function (default: MSE).')
    parser.add_argument(
        '--predictor_transform',
        type=str,
        default='0',
        metavar='str',
        help='Comma-delimited list representing which transformation to apply to which input parameter within a configuration (default: 0).')
    parser.add_argument(
        '--reg',
        type=str,
        default='1e-5',
        metavar='str',
        help='regularization parameter (default: 1e-5).')
    parser.add_argument(
        '--max_spline_degrees',
        type=str,
        default="3",
        metavar='str',
        help='Comma-delimited list of number of maximum spline degrees degrees (default: 3).')
    parser.add_argument(
        '--tree_depth',
        type=str,
        default="8",
        metavar='str',
        help='Comma-delimited list representing maximum decision tree depth (default: 8).')
    parser.add_argument(
        '--ntrees',
        type=str,
        default="16",
        metavar='str',
        help='Comma-delimited list representing number of weak decision tree learners (default: 16).')
    parser.add_argument(
        '--nneighbors',
        type=str,
        default="5",
        metavar='str',
        help='Comma-delimited list representing number of neighbors (default: 5).')
    parser.add_argument(
        '--nlevels',
        type=str,
        default='3',
        metavar='str',
        help='Comma-delimited list of number of sparse grid levels (default: 3).')
    parser.add_argument(
        '--nadaptpts',
        type=str,
        default='3',
        metavar='str',
        help='Comma-delimited list of number of grid-points to update (default: 3).')
    parser.add_argument(
        '--nrefinements',
        type=int,
        default='5',
        metavar='int',
        help='Number of sparse-grid refinements (default: 5).')
    parser.add_argument(
        '--hidden_layer_sizes',
        type=str,
        default='64,64',
        metavar='str',
        help='Comma-delimited list signifying the number of units in each layer of a MLP (default: 64,64).')
    parser.add_argument(
        '--activation',
        type=str,
        default='relu',
        metavar='str',
        help='Activation function for a MLP (default: relu).')
    parser.add_argument(
        '--solver',
        type=str,
        default='adam',
        metavar='str',
        help='Solver for optimizing a MLP (default: adam)')
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
