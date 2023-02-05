def add_general_arguments(parser):

    parser.add_argument(
        '--niter',
        type=int,
        default=0,
        metavar='int',
        help='Number of executions per input tuple (default: 0)')
    parser.add_argument(
        '--set_size',
        type=int,
        default=0,
        metavar='int',
        help='Number of randomly sampled inputs constrained to grid-points (default: 0)')
    parser.add_argument(
        '--sample_type',
        type=int,
        default='0',
        metavar='int',
        help='ID for placement of grid-points constrained to a particular discretization granularity as specified by grid_type. Equivalently, ID for sampling distribution (default: 0)')
    parser.add_argument(
        '--thread_count',
        type=str,
        default='1',
        metavar='str',
        help='thread counts to include along tensor mode ID')
    parser.add_argument(
        '--scale_mode',
        type=str,
        default='strong',
        metavar='str',
        help='specification of weak scaling or strong scaling')
    parser.add_argument(
        '--kernel_type',
        type=int,
        default=0,
        metavar='int',
        help='ID specifying task, hardware, and/or tuning parameters to evaluate (default: 0)')
    parser.add_argument(
        '--print_header',
        type=int,
        default=0,
        metavar='int',
        help='ID specifying whether to print the header of index column names (default: 0)')
