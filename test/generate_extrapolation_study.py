def generate_experiments(args):
    job_script_info="\
#!/bin/bash\n\
#SBATCH -J %s\n\
#SBATCH -o %s.o\n\
#SBATCH -e %s.e\n\
#SBATCH -p skx-normal\n\
#SBATCH -N 1\n\
#SBATCH -n 1\n\
#SBATCH -t 04:00:00\n\n\
    "%(args.output_file,args.output_file,args.output_file)

    default_hyperparameters="\
export CPPM_VERBOSE=0\n\
export CPPMI_PARTITION_SPACING=GEOMETRIC\n\
export CPPME_PARTITION_SPACING=GEOMETRIC\n\
export CPPMI_PARTITIONS_PER_DIMENSION=64\n\
export CPPME_PARTITIONS_PER_DIMENSION=64\n\
export CPPMI_OBS_PER_PARTITION=512\n\
export CPPME_OBS_PER_PARTITION=512\n\
export CPPMI_CP_RANK=3\n\
export CPPME_CP_RANK=1\n\
export CPPMI_RUNTIME_TRANSFORM=LOG\n\
export CPPME_RUNTIME_TRANSFORM=NONE\n\
export CPPMI_MAX_SPACING_FACTOR=2\n\
export CPPME_MAX_SPACING_FACTOR=2\n\
export CPPMI_LOSS_FUNCTION=MSE\n\
export CPPME_LOSS_FUNCTION=MLogQ2\n\
export CPPMI_REGULARIZATION=1e-8\n\
export CPPME_REGULARIZATION=1e-6\n\
export CPPMI_OPTIMIZATION_BARRIER_START=1e-1\n\
export CPPME_OPTIMIZATION_BARRIER_START=1e-1\n\
export CPPMI_OPTIMIZATION_BARRIER_STOP=1e-11\n\
export CPPME_OPTIMIZATION_BARRIER_STOP=1e-11\n\
export CPPMI_OPTIMIZATION_BARRIER_REDUCTION_FACTOR=8\n\
export CPPME_OPTIMIZATION_BARRIER_REDUCTION_FACTOR=8\n\
export CPPMI_FM_CONVERGENCE_TOL=1e-3\n\
export CPPME_FM_CONVERGENCE_TOL=1e-3\n\
export CPPMI_FM_MAX_NUM_ITER=10\n\
export CPPME_FM_MAX_NUM_ITER=10\n\
export CPPMI_MAX_NUM_SWEEPS=4\n\
export CPPME_MAX_NUM_SWEEPS=4\n\
export CPPMI_SWEEP_TOL=1e-2\n\
export CPPME_SWEEP_TOL=1e-2\n\
export CPPMI_MAX_NUM_RE_INITS=1\n\
export CPPME_MAX_NUM_RE_INITS=1\n\
export CPPMI_MIN_NUM_OBS_FOR_TRAINING=64\n\
export CPPME_MIN_NUM_OBS_FOR_TRAINING=64\n\
export CPPMI_OPTIMIZATION_CONVERGENCE_TOLERANCE_FOR_RE_INIT=1e-1\n\
export CPPME_OPTIMIZATION_CONVERGENCE_TOLERANCE_FOR_RE_INIT=1e-1\n\
export CPPMI_INTERPOLATION_FACTOR_TOL=0.5\n\
export CPPME_INTERPOLATION_FACTOR_TOL=0.5\n\
export CPPMI_AGGREGATE_OBS_ACROSS_COMM=0\n\
export CPPME_AGGREGATE_OBS_ACROSS_COMM=0\n\
export CPPME_MAX_SPLINE_DEGREE=1\n\
export CPPME_MAX_TRAINING_SET_SIZE=16\n\
export CPPME_FACTOR_MATRIX_ELEMENT_TRANSFORM=NONE\n\
export CPPME_FACTOR_MATRIX_UNDERLYING_POSITION_TRANSFORM=NONE\n\
export CPPME_USE_SPLINE=0\n\n\
"

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
    my_file = open("%s.sh"%(args.output_file),'w')
    my_file.write(job_script_info)
    my_file.write(default_hyperparameters)

    for count,hyperparameters in enumerate(configurations):
        my_file.write("# Exp %d\n"%(count))
        my_file.write("CPPME_CP_RANK=%d\n"%(hyperparameters[0]))
        my_file.write("CPPME_REGULARIZATION=%g\n"%(hyperparameters[1]))
        my_file.write("CPPME_PARTITIONS_PER_DIMENSION=%d\n"%(hyperparameters[2]))
        my_file.write("CPPME_MAX_SPLINE_DEGREE=%d\n"%(hyperparameters[3]))
        my_file.write("CPPME_FACTOR_MATRIX_ELEMENT_TRANSFORM=%s\n"%(hyperparameters[4]))
        my_file.write("CPPME_FACTOR_MATRIX_UNDERLYING_POSITION_TRANSFORM=%s\n"%(hyperparameters[5]))
        my_file.write("CPPME_MAX_TRAINING_SET_SIZE=%d\n"%(hyperparameters[6]))
        my_file.write("ibrun -n 1 dgemm3 %s %s %s.csv\n"%(args.training_set_file,args.test_set_file,args.output_file))

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i',
        '--training-set-file',
        type=str,
        required=True,
        help='Path to training set file.')
    parser.add_argument(
        '-j',
        '--test-set-file',
        type=str,
        required=True,
        help='Path to test set file.')
    parser.add_argument(
        '-k',
        '--output-file',
        type=str,
        required=True,
        help='Path to output file that experimental results are written to.')
    args, _ = parser.parse_known_args()
    generate_experiments(args)
