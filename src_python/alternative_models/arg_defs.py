def add_general_arguments(parser):

    parser.add_argument(
        '--write-header',
        type=int,
        default='0',
        help='Boolean decision whether to write column headers to CSV file (default: 0).')
    parser.add_argument(
        '--training-set-size',
        type=int,
        default=0,
        help='Size of training set (default: 0).')
    parser.add_argument(
        '--test-set-size',
        type=int,
        default=0,
        help='Size of test set (default: 0).')
    parser.add_argument(
        '--training-set-split-percentage',
        type=float,
        default='0',
        help='Percentage of the training set used for model selection across hyper-parameter space (default: 0).')
    parser.add_argument(
        '--response-transform',
        type=int,
        default="1",
        help='Transformation to apply to runtime data (default: 1 (Log transformation)).')
    parser.add_argument(
        '--error-metric',
        type=str,
        default='MSE',
        help='Error metric characterizing loss function (default: MSE).')
    parser.add_argument(
        '--predictor-transform',
        type=str,
        default='0',
        help='Comma-delimited list representing which transformation to apply to which input parameter within a configuration (default: 0).')
    parser.add_argument(
        '--reg',
        type=str,
        default='1e-5',
        help='regularization parameter (default: 1e-5).')
    parser.add_argument(
        '--max-spline-degrees',
        type=str,
        default="3",
        help='Comma-delimited list of number of maximum spline degrees degrees (default: 3).')
    parser.add_argument(
        '--tree-depth',
        type=str,
        default="8",
        help='Comma-delimited list representing maximum decision tree depth (default: 8).')
    parser.add_argument(
        '--ntrees',
        type=str,
        default="16",
        help='Comma-delimited list representing number of weak decision tree learners (default: 16).')
    parser.add_argument(
        '--training-file',
        type=str,
        default='',
        help='File path to training dataset.')
    parser.add_argument(
        '--test-file',
        type=str,
        default='',
        help='File path to test dataset.')
    parser.add_argument(
        '--output-file',
        type=str,
        default='',
        help='File path to write prediction results.')
    parser.add_argument(
        '--input-columns',
        type=str,
        default='',
        help='Comma-delimited list of column indices corresponding to each parameter witin a configuration for both training and test datasets.')
    parser.add_argument(
        '--data-columns',
        type=str,
        default='',
        help='Comma-delimited list of column indices corresponding to response (execution time) within training and test datasets (same format assumed for both).')
    parser.add_argument(
        '--mode-range-min',
        type=str,
        default='',
        help='Comma-delimited list of minimum values for each parameter within a configuration.')
    parser.add_argument(
        '--mode-range-max',
        type=str,
        default='',
        help='Comma-delimited list of maximum values for each parameter within a configuration.')
    parser.add_argument(
        '--print-model-parameters',
        type=int,
        default=0,
        help='Whether or not to print the factor matrix elements (default: 0).')
    parser.add_argument(
        '-v',
        '--verbose',
        type=int,
        default=0,
        help='Whether or not to print information about datasets (default: 0).')
    parser.add_argument(
        '--print-test-error',
        type=int,
        default="0",
        help='')
