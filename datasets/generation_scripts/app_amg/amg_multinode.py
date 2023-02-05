import time, sys
import pyearth as pe
import numpy as np
import numpy.linalg as la
import scipy.interpolate as sci
import scipy.optimize as sco
import scipy.sparse.linalg as sla
from ctf import random as crandom
import random as rand
import os
import argparse
import arg_defs as arg_defs

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

    kernel_name = "amg"
    write_file_location = "./"
    ndim=5
    mode_range_min = [16,16,16,1,1]
    mode_range_max = [128,128,128,64,1024]
    sampling_distribution = [1,1,1,1,1]#[int(n) for n in args.cell_spacing_mode.split(',')]
    grid_str_suffix = []
    alg_str = "amg_kt0"
    ppn_count=[1,2,4,8,16,32,64]
    thread_count = [1,2,4,8,16,32,64]
    nnodes=[1]
    core_count =64
    coarsening_type_list = [0,3,6,8,10,21,22]
    relax_type_list = [0,3,4,6,8,13,14,16,17,18]
    interp_type_list = [0,2,3,4,5,6,8,9,12,13,14,16,17,18]

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
		write_files[-1].write("#!/bin/bash\n#SBATCH -J %s\n#SBATCH -o %s.o\n#SBATCH -e %s.e\n#SBATCH -p normal\n#SBATCH -N %d\n#SBATCH -n %d\n#SBATCH -t 04:00:00\nexport OMP_NUM_THREADS=%d\nexport AMG_NODE_COUNT=%d\nexport AMG_PPN_COUNT=%d\nexport AMG_KERNEL_TYPE=%d\n\n"%(kernel_tag,kernel_tag,kernel_tag,nnodes[i],ppn_count[j]*nnodes[i],thread_count[k],nnodes[i],ppn_count[j],args.kernel_type))
		file_dict[(nnodes[i],ppn_count[j],thread_count[k])] = valid_file_count
		valid_file_count += 1

    j=0
    while (j<args.set_size):
	ppn_idx = np.random.randint(0,len(ppn_count))
	ppn = ppn_count[ppn_idx]
	thread_idx = np.random.randint(0,len(thread_count))
	thread = thread_count[thread_idx]
	node_idx = np.random.randint(0,len(nnodes))
	node = nnodes[node_idx]
        if (ppn*thread > 128 or ppn*thread < 64):
            continue

        nx = generate_sample(mode_range_min[0],mode_range_max[0],sampling_distribution[0],1,2)
        ny = generate_sample(mode_range_min[1],mode_range_max[1],sampling_distribution[1],1,2)
        nz = generate_sample(mode_range_min[2],mode_range_max[2],sampling_distribution[2],1,2)
        _input_mode1 = nx
        _input_mode2 = ny
        _input_mode3 = nz
        process_count = node*ppn
        px=1
        py=1
        pz=1
        while (process_count>1):
            if (process_count == 3):
		if (_input_mode1 >= _input_mode2 and _input_mode1 >= _input_mode3):
		    px *= 3
		    _input_mode1 /= 3
		elif (_input_mode2 > _input_mode1 and _input_mode2 >= _input_mode3):
		    py *= 3
		    _input_mode2 /= 3
		elif (_input_mode3 > _input_mode1 and _input_mode3 > _input_mode2):
		    pz *= 3
		    _input_mode3 /= 3
		else:
		    assert(0)
                process_count /= 3
            else:
		if (_input_mode1 >= _input_mode2 and _input_mode1 >= _input_mode3):
		    px *= 2
		    _input_mode1 /= 2
		elif (_input_mode2 > _input_mode1 and _input_mode2 >= _input_mode3):
		    py *= 2
		    _input_mode2 /= 2
		elif (_input_mode3 > _input_mode1 and _input_mode3 > _input_mode2):
		    pz *= 2
		    _input_mode3 /= 2
		else:
		    assert(0)
                process_count /= 2
	# Write this (below) before input_mode1,input_mode2,input_mode3 gets overwritten
        assert(px*py*pz == node*ppn)
	tp1_idx = np.random.randint(0,len(coarsening_type_list))
	tp1 = coarsening_type_list[tp1_idx]
	tp2_idx = np.random.randint(0,len(relax_type_list))
	tp2 = relax_type_list[tp2_idx]
	tp3_idx = np.random.randint(0,len(interp_type_list))
	tp3 = interp_type_list[tp3_idx]
	write_files[file_dict[(nnodes[node_idx],ppn,thread)]].write("ibrun -n %d ./amg -n %d %d %d -P %d %d %d -coarsen_type %d -relax_type %d -interp_type %d\n"%(node*ppn,nx,ny,nz,px,py,pz,tp1,tp2,tp3))
        j += 1
