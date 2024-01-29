def add_shared_parameters(parser):
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true')
    parser.add_argument(
        '-W',
        '--write-header',
        action='store_true',
        help='Write row of strings describing each output column to output file.')
    parser.add_argument(
        '-T',
        '--training-set-size',
        type=int,
        default=-1,
        help='Number of observations from provided dataset (sampled randomly) used to train model. Negative default value signifies to use provided dataset directly without sampling.')
    parser.add_argument(
        '-U',
        '--test-set-size',
        type=int,
        default=-1,
        help='Number of observations from provided dataset (sampled randomly) used to test the trained model. Negative default value signifies to use provided dataset directly without sampling.')
    parser.add_argument(
        '-A',
        '--response-transform-type',
        type=int,
        default=1,
        help='Transformation to apply to execution times in provided dataset data.')
    parser.add_argument(
        '-i',
        '--training-file',
        type=str,
        required=True,
        help='File path to dataset used to train model.')
    parser.add_argument(
        '-j',
        '--test-file',
        type=str,
        required=True,
        help='File path to dataset used to test model.')
    parser.add_argument(
        '-o',
        '--output-file',
        type=str,
        required=True,
        help='File path to output data.')
    parser.add_argument(
        '-g',
        '--input-columns',
        type=int,
        nargs='+',
        required=True,
        help='Indices of columns corresponding to each parameter within a configuration for all datasets.')
    parser.add_argument(
        '-d',
        '--data-columns',
        type=int,
        nargs='+',
        required=True,
        help='Indices of columns corresponding to execution times of configurations for all datasets.')
    parser.add_argument(
        '-Z',
        '--print-test-error',
        action='store_true',
        help='Print error of model applied to provided test dataset.')
    parser.add_argument(
        '-M',
        '--mode-range-min',
        type=float,
        nargs='+',
        default=[],
        help='Minimum range for each parameter')
    parser.add_argument(
        '-N',
        '--mode-range-max',
        type=float,
        nargs='+',
        default=[],
        help='Maximum range for each parameter.')
    parser.add_argument(
        '-P',
        '--predictor-transform-type',
        type=int,
        nargs='+',
        required=True,
        help='Signifies the transformation to apply to each parameter: no-op (0) or logarithmic (1).')
    parser.add_argument(
        '-Y',
        '--print-model-parameters',
        action='store_true',
        help='Print factor matrix elements.')
    parser.add_argument(
        '--train-range-start-idx',
        type=int,
        default=-1,
        help='Start index for consideration in dataset.')
    parser.add_argument(
        '--train-range-end-idx',
        type=int,
        default=-1,
        help='End index for consideration in dataset.')
    parser.add_argument(
        '--test-range-start-idx',
        type=int,
        default=-1,
        help='Start index for consideration in dataset.')
    parser.add_argument(
        '--test-range-end-idx',
        type=int,
        default=-1,
        help='End index for consideration in dataset.')
