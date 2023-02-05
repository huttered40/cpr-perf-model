def add_general_arguments(parser):

    parser.add_argument(
        '--set_size',
        type=int,
        default=0,
        metavar='int',
        help='Number of randomly sampled inputs constrained to grid-points (default: 0)')
    parser.add_argument(
        '--kernel_type',
        type=int,
        default=0,
        metavar='int',
        help='ID specifying task, hardware, and/or tuning parameters to evaluate (default: 0)')
