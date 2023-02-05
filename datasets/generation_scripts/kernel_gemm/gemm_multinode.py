import time, sys
import numpy as np
import random as rand
import os
import argparse
import arg_defs as arg_defs

def generate_sample(low,high,_sample_mode,size=1,base=np.e):
    if (_sample_mode == 0):
	return rand.randint(low,high-1)
    elif (_sample_mode == 1):
	return int(loguniform(low,high,size,base))

def loguniform(low, high, size, base=np.e):
    return np.power(base, np.random.uniform(np.log2(low), np.log2(high), size))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    arg_defs.add_general_arguments(parser)
    args, _ = parser.parse_known_args()

    """
    kernel_type signifies the following:
      1. the domain from which the dataset is generated.
      2. how noisy each sample is
    """

    num_nodes = 16
    kernel_name = "gemm"
    write_file_location = "/work2/05608/tg849075/cpr-perf-model/datasets/generation_scripts/kernel_gemm/"
    thread_count = [int(n) for n in args.thread_count.split(',')]
    if (args.kernel_type%10 == 0):
        # Executions on a single core
        mode_range_min = [32,32,32]
        mode_range_max = [4096,4096,4096]
        restrict = 4096
        binary_name = "gemm" if args.kernel_type == 0 else "gemm_noisy"
    elif (args.kernel_type%10 == 1):
        # Executions on a single core
        mode_range_min = [32,32,32]
        mode_range_max = [524288,524288,524288]
        restrict = 4096
        binary_name = "gemm" if args.kernel_type == 1 else "gemm_noisy"
    else:
        assert(0)
    sampling_distribution = [args.sample_type,args.sample_type,args.sample_type]

    kernel_tag = "%s_kt%d_st%d"%(kernel_name,args.kernel_type,args.sample_type)
    write_file = []
    for j in range(num_nodes):
        for i in thread_count:
	    file_str = '%s_%dthreads_node%d.sh'%(kernel_tag,i,j+1)
	    file_str = write_file_location + file_str
	    write_file.append(open(file_str,"a"))
	    write_file[-1].write("#!/bin/bash\n#SBATCH -J %s\n#SBATCH -o %s.o\n#SBATCH -e %s.e\n#SBATCH -p normal\n#SBATCH -N 1\n#SBATCH -n 1\n#SBATCH -t 08:00:00\nexport MKL_NUM_THREADS=%d\n\n"%(kernel_tag,kernel_tag,kernel_tag,i))

    rand.seed()
    for i in range(args.set_size):
	is_legal = False
	while (is_legal == False):
	    input_mode4_idx = generate_sample(0,len(thread_count),0,1,2)
	    input_mode4 = thread_count[input_mode4_idx]
	    input_mode1 = np.sqrt(input_mode4)*generate_sample(mode_range_min[0],mode_range_max[0],sampling_distribution[0],1,2)
	    input_mode2 = np.sqrt(input_mode4)*generate_sample(mode_range_min[1],mode_range_max[1],sampling_distribution[1],1,2)
	    input_mode3 = np.sqrt(input_mode4)*generate_sample(mode_range_min[2],mode_range_max[2],sampling_distribution[2],1,2)
            if (args.kernel_type==1 and (input_mode1*input_mode2 + input_mode1*input_mode3 + input_mode2*input_mode3) > restrict**2):
		is_legal = False
            else:
		is_legal = True
	input_tuple = (input_mode1,input_mode2,input_mode3,input_mode4)
        write_file[(i%num_nodes)*len(thread_count)+input_mode4_idx].write("/work2/05608/tg849075/qmac/training_sets/generate/gemm_multinode %d %d %d %d %d %d %d %d\n"%(args.kernel_type,args.sample_type,input_tuple[0],input_tuple[1],input_tuple[2],input_tuple[3],i,i%num_nodes))
