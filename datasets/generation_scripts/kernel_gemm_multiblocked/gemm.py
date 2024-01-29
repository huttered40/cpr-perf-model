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

    machine_name = "icx" # {"knl", "skx", "icx"}
    kernel_name = "multi-blocked-gemm"
    write_file_location = "./"
    nhours = 8
    node_count = 32
    register_block_size_m = [1,2,4,8]
    register_block_size_n = [1,2,4,8]
    register_block_size_k = [2,4,8]
    outer_cache_block_size = range(64,513,8)
    inner_cache_block_size = range(8,257,8)
    write_file_location = "/work2/05608/tg849075/cpr-perf-model/datasets/generation_scripts/kernel_gemm_multiblocked/"
    sampling_distribution = [args.sample_type,args.sample_type,args.sample_type]


    if (args.kernel_type%10 == 0):
        mode_range_min = [32,32,32]
        mode_range_max = [4096,4096,4096]
        restrict = 4096
    elif (args.kernel_type%10 == 1):
        mode_range_min = [128,128,128]
        mode_range_max = [524288,524288,524288]
        restrict = 8192
    elif (args.kernel_type%10 == 2):
        mode_range_min = [2048,2048,2048]
        mode_range_max = [2049,2049,2049]
        restrict = 8192
    else:
        assert(0)

    if (machine_name == "knl"):
        thread_count = [32,64,128,256]
        core_count = 64
        queue_name = "normal"
        arch_flag_name = "knl"
    elif (machine_name == "skx"):
        thread_count = [24,48,96]
        core_count = 48
        queue_name = "skx-normal"
        arch_flag_name = "skylake"
    elif (machine_name == "icx"):
        thread_count = [40,80,160]
        core_count = 80
        queue_name = "icx-normal"
        arch_flag_name = "icelake-server"
    instruction_scheduling_flag = ["","-march=%s"%(arch_flag_name)]

    write_files = []
    file_dict = {}
    valid_file_count = 0
    for i in range(node_count):
	for j in range(len(thread_count)):
	    kernel_tag = "%s_%s_kt%d_%s_thread%d_nodeID%d"%(kernel_name,machine_name,args.kernel_type,machine_name,thread_count[j],i)
	    file_str = '%s.sh'%(kernel_tag)
	    file_str = write_file_location + file_str
	    write_files.append(open(file_str,"w"))
	    write_files[-1].write("#!/bin/bash\n#SBATCH -J %s\n#SBATCH -o %s.o\n#SBATCH -e %s.e\n#SBATCH -p %s\n#SBATCH -N %d\n#SBATCH -n %d\n#SBATCH -t 0%d:00:00\nmodule load gcc/9.1.0\nexport OMP_NUM_THREADS=%d\nexport MACHINE_NAME=%s\n\n"%(kernel_tag,kernel_tag,kernel_tag,queue_name,1,1,nhours,thread_count[j],machine_name))
	    file_dict[(thread_count[j],i)] = valid_file_count
	    valid_file_count += 1

    rand.seed()
    for i in range(args.set_size):
	is_legal = False
	while (is_legal == False):
	    input_mode1 = generate_sample(mode_range_min[0],mode_range_max[0],sampling_distribution[0],1,2)
	    input_mode2 = generate_sample(mode_range_min[1],mode_range_max[1],sampling_distribution[1],1,2)
	    input_mode3 = generate_sample(mode_range_min[2],mode_range_max[2],sampling_distribution[2],1,2)
            if (input_mode1==2049 or input_mode2==2049 or input_mode3==2049):
                continue
	    tp1m = outer_cache_block_size[np.random.randint(0,len(outer_cache_block_size))]
	    tp1n = outer_cache_block_size[np.random.randint(0,len(outer_cache_block_size))]
	    tp1k = outer_cache_block_size[np.random.randint(0,len(outer_cache_block_size))]
	    tp2m = inner_cache_block_size[np.random.randint(0,len(inner_cache_block_size))]
	    tp2n = inner_cache_block_size[np.random.randint(0,len(inner_cache_block_size))]
	    tp2k = inner_cache_block_size[np.random.randint(0,len(inner_cache_block_size))]
	    tp3m = register_block_size_m[np.random.randint(0,len(register_block_size_m))]
	    tp3n = register_block_size_n[np.random.randint(0,len(register_block_size_n))]
	    tp3k = register_block_size_k[np.random.randint(0,len(register_block_size_k))]
	    tp4 = instruction_scheduling_flag[np.random.randint(0,len(instruction_scheduling_flag))]
            if ((args.kernel_type%10)==1 and (input_mode1*input_mode2 + input_mode1*input_mode3 + input_mode2*input_mode3) > restrict**2):
		is_legal = False
                continue
            if (not(tp3m <= tp2m and tp2m <= tp1m and tp1m <= input_mode1)):
		is_legal = False
                continue
            if (not(tp3n <= tp2n and tp2n <= tp1n and tp1n <= input_mode2)):
		is_legal = False
                continue
            if (not(tp3k <= tp2k and tp2k <= tp1k and tp1k <= input_mode3)):
		is_legal = False
                continue
            is_legal = True
        assert(tp3m <= tp2m and tp2m <= tp1m and tp1m <= input_mode1)
        assert(tp3n <= tp2n and tp2n <= tp1n and tp1n <= input_mode2)
        assert(tp3k <= tp2k and tp2k <= tp1k and tp1k <= input_mode3)
	tp5 = thread_count[np.random.randint(0,len(thread_count))]
        node_id = np.random.randint(0,node_count)
        write_files[file_dict[(tp5,node_id)]].write("g++ %s -fopenmp -O3 -mavx512f -mavx512cd -mavx512vl -DRBM=%d -DRBN=%d -DRBK=%d multi-blocked-gemm.cpp -o multi-blocked-gemm_%dthreads_%dnodeID\n"%(tp4,tp3m,tp3n,tp3k,tp5,node_id))
        write_files[file_dict[(tp5,node_id)]].write("./multi-blocked-gemm_%dthreads_%dnodeID %d %d %d %d %d %d %d %d %d %d %d 0\n"%(tp5,node_id,input_mode1,input_mode2,input_mode3,tp1m,tp1n,tp1k,tp2m,tp2n,tp2k,tp5,node_id))
