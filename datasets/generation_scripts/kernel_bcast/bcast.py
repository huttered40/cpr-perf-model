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

    kernel_name = "bcast"
    write_file_location = "/work2/05608/tg849075/cpr-perf-model/datasets/generation_scripts/kernel_bcast/"
    if (args.kernel_type == 0):
        node_count = [1,2,4,8,16,32,64,128,256,512,1024]
        ppn_count = [1,2,4,8,16,32,64]
        mode_range_min = [2**10,1,1]
        mode_range_max = [2**24,64,1024]
        sampling_distribution = [args.sample_type,1,1]
    elif (args.kernel_type == 1):
        node_count = [1,2,4,8,16,32,64,128,256,512,1024]
        ppn_count = [1,2,4,8,16,32,64]
        mode_range_min = [2**16,1,1]
        mode_range_max = [2**26,64,1024]
        sampling_distribution = [args.sample_type,1,1]
    else:
        assert(0)

    write_files = []
    for i in range(len(node_count)):
        for j in range(len(ppn_count)):
	    kernel_tag = "%s_kt%d_st%d_np%d_ppn%d"%(kernel_name,args.kernel_type,args.sample_type,node_count[i],ppn_count[j])
	    file_str = '%s.sh'%(kernel_tag)
	    file_str = write_file_location + file_str
	    write_files.append(open(file_str,"a"))
	    write_files[-1].write("#!/bin/bash\n#SBATCH -J %s\n#SBATCH -o %s.o\n#SBATCH -e %s.e\n#SBATCH -p normal\n#SBATCH -N %d\n#SBATCH -n %d\n#SBATCH -t 04:00:00\nexport OMP_NUM_THREADS=%d\n\n"%(kernel_tag,kernel_tag,kernel_tag,node_count[i],ppn_count[j]*node_count[i],1))

    for i in range(args.set_size):
	input_mode1 = generate_sample(mode_range_min[0],mode_range_max[0],sampling_distribution[0],1,2)
	ppn_idx = rand.randint(0,len(ppn_count)-1)
	input_mode2 = ppn_count[ppn_idx]
	node_idx = rand.randint(0,len(node_count)-1)
	input_mode3 = node_count[node_idx]
	input_tuple = (input_mode1,input_mode2,input_mode3)
	write_files[node_idx*len(ppn_count)+ppn_idx].write("ibrun -n %d %s %d %d %d %d %d %d\n"%(input_mode2*input_mode3,kernel_name,args.kernel_type,args.sample_type,input_tuple[0],input_tuple[1],input_tuple[2],i))
