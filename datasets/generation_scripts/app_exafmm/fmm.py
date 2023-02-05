import time, sys
import pyearth as pe
import numpy as np
import numpy.linalg as la
import scipy.interpolate as sci
import scipy.optimize as sco
import scipy.sparse.linalg as sla
import random as rand
import gzip
import shutil
import os
import argparse
import arg_defs as arg_defs

def round_up(n):
    if n==0:
        return 1
    elif n<=2:
        return 2
    elif n<=4:
        return 4
    elif n<=8:
        return 8
    elif n<=16:
        return 16

def loguniform(low, high, size, base=np.e):
    return np.power(base, np.random.uniform(np.log2(low), np.log2(high), size))

def generate_sample(low,high,_sample_mode,size=1,base=np.e):
    if (_sample_mode == 0):
	return rand.randint(low,high-1)
    elif (_sample_mode == 1):
	return int(loguniform(low,high,size,base))
    elif (_sample_mode == 2):
	return int(invuniform(low,high,size))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    arg_defs.add_general_arguments(parser)
    args, _ = parser.parse_known_args()

    kernel_name = "exafmm"
    write_file_location = "./"
    core_count=64
    ppn_count=[1,2,4,8,16,32,64]
    thread_count = [1,2,4,8,16,32,64]
    nnodes=[1]#,2,4,8,16,32,64,128,256,512]
    dist = ['l','c','s','o','p']
    # nnodes, ppn, nthreads, nbodies, ncrit, level, order
    mode_range_min = [1,1,1,4096,32,1,4]
    mode_range_max = [1,1,64,65536,256,5,16]
    sampling_distribution = [1,1,1,1,0,0,0]

    # Do not include training set size in file name. The idea is that I will populate this file with a max number of training samples
    #   and take subsets from it.
    # The goal is to make these file names independent of the dimension of the kernel.
    write_files = []
    file_dict = {}
    valid_file_count = 0
    for i in range(len(nnodes)):
	for j in range(len(ppn_count)):
	    for k in range(len(thread_count)):
                if (ppn_count[j]*thread_count[k]>128 or ppn_count[j]*thread_count[k]<64):
                    continue
		kernel_tag = "%s_kt%d_ppn%d_thread%d_node%d"%(kernel_name,args.kernel_type,ppn_count[j],thread_count[k],nnodes[i])
		file_str = '%s.sh'%(kernel_tag)
		file_str = write_file_location + file_str
		write_files.append(open(file_str,"w"))
		write_files[-1].write("#!/bin/bash\n#SBATCH -J %s\n#SBATCH -o %s.o\n#SBATCH -e %s.e\n#SBATCH -p normal\n#SBATCH -N %d\n#SBATCH -n %d\n#SBATCH -t 04:00:00\nexport OMP_NUM_THREADS=%d\nexport FMM_NUM_NODES=%d\nexport FMM_PPN=%d\nexport FMM_KT=%d\n\n"%(kernel_tag,kernel_tag,kernel_tag,nnodes[i],ppn_count[j]*nnodes[i],thread_count[k],nnodes[i],ppn_count[j],args.kernel_type))
		file_dict[(nnodes[i],ppn_count[j],thread_count[k])] = valid_file_count
		valid_file_count += 1

    j=0
    while (j<args.set_size):
	ppn_idx = np.random.randint(0,len(ppn_count))
	ppn = ppn_count[ppn_idx]
	thread_idx = np.random.randint(0,len(thread_count))
	thread = thread_count[thread_idx]
        if (ppn*thread > 128 or ppn*thread < 64):
            continue
        nb = generate_sample(mode_range_min[3]/ppn,mode_range_max[3]/ppn,sampling_distribution[3],1,2)
        nc = generate_sample(mode_range_min[4],mode_range_max[4],sampling_distribution[4],1,2)
        l = generate_sample(mode_range_min[5],mode_range_max[5],sampling_distribution[5],1,2)
        o = generate_sample(mode_range_min[6],mode_range_max[6],sampling_distribution[6],1,2)
        d = 'c'#dist[np.random.randint(0,len(dist))]
	node_idx = np.random.randint(0,len(nnodes))
	node = nnodes[node_idx]
	write_files[file_dict[(nnodes[node_idx],ppn,thread)]].write("ibrun -n %d /work2/05608/tg849075/exafmm/tests/laplace --numBodies %d --distribution %c --level %d --ncrit %d --P %d\n"%(ppn*nnodes[node_idx],nb,d,l,nc,o))
        j += 1
