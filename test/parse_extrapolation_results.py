def generate_experiments(args):

    """
    cp_rank = [1,2,3,4]
    regularization = [0,1e-8,1e-7,1e-6]
    tensor_mode_length = [16,32,64]
    max_spline_degree = [1,2]
    runtime_transform = ["NONE"]#,"LOG"]
    midpoint_transform = ["NONE"]#,"LOG"]
    max_training_set_size = [8,16,32,64]

    configurations = []
    for i in cp_rank:
        for j in regularization:
            for k in tensor_mode_length:
                for l in max_spline_degree:
                    for m in runtime_transform:
                        for n in midpoint_transform:
                            for o in max_training_set_size:
                                configurations.append([i,j,k,l,m,n,o])
    print(args)
    """
    my_file = open("%s"%(args.input_file),'r')
    lines = my_file.readlines()
    errors = []
    for line in lines:
        data = line.split(',')
        errors.append(float(data[17]))

    errors.sort()
    print(errors)
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i',
        '--input-file',
        type=str,
        required=True,
        help='Path to file that experimental results are stored.')
    args, _ = parser.parse_known_args()
    generate_experiments(args)
