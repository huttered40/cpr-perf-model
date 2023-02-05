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

def loguniform(low, high, size, base=np.e):
    return np.power(base, np.random.uniform(np.log2(low), np.log2(high), size))

def generate_sample(low,high,_sample_mode,size=1,base=np.e):
    if (_sample_mode == 0):
	return rand.randint(low,high-1)
    elif (_sample_mode == 1):
	return int(loguniform(low,high,size,base))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    arg_defs.add_general_arguments(parser)
    args, _ = parser.parse_known_args()

    kernel_name = "kripke"
    write_file_location = "/work2/05608/tg849075/cpr-perf-model/datasets/generation_scripts/app_kripke/"
    core_count=64
    ppn_count=[1,2,4,8,16,32,64]
    thread_count = [1,2,4,8,16,32,64]
    nnodes=[1]#,2,4,8,16,32,64,128,256,512]
    if (args.kernel_type == 0):
        # Executions on a single core
        mode_range_max = [128,4,128,256]
        # groups
        legendre = np.arange(0,5)
        # quad/direction
        # zones
        layout = ["DGZ","DZG","GDZ","GZD","ZDG","ZGD"]
        dset = [8,16,24,32,40,48,56,64]
        gset = [1,2,4,8,16,32]
        #zset = [1,2,4,8,16]
        solve = ["sweep","bj"]
        _zonex = 32
        _zoney = 32
        _zonez = 32
    else:
        assert(0)

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
		write_files[-1].write("#!/bin/bash\n#SBATCH -J %s\n#SBATCH -o %s.o\n#SBATCH -e %s.e\n#SBATCH -p normal\n#SBATCH -N %d\n#SBATCH -n %d\n#SBATCH -t %d:00:00\nexport OMP_NUM_THREADS=%d\nexport KRIPKE_IS_TEST=0\nexport KRIPKE_NUM_NODES=%d\nexport KRIPKE_PPN=%d\n\n"%(kernel_tag,kernel_tag,kernel_tag,nnodes[i],ppn_count[j]*nnodes[i],10-int(np.log2(nnodes[i])),thread_count[k],nnodes[i],ppn_count[j]))
		file_dict[(nnodes[i],ppn_count[j],thread_count[k])] = valid_file_count
		valid_file_count += 1

    config_dict = {}
    group_dict = {}
    quad_dict = {}
    j=0
    while (j<args.set_size):
	ppn_idx = np.random.randint(0,len(ppn_count))
	ppn = ppn_count[ppn_idx]
	thread_idx = np.random.randint(0,len(thread_count))
	thread = thread_count[thread_idx]
        if (ppn*thread > 128 or ppn*thread < 64):
            continue
        _layout = layout[np.random.randint(0,len(layout))]
        _solve = solve[np.random.randint(0,len(solve))]
        _legendre = legendre[np.random.randint(0,len(legendre))]
        _dset = dset[np.random.randint(0,len(dset))]
        _gset = gset[np.random.randint(0,len(gset))]
        #_zsetx = zset[np.random.randint(0,len(zset))]
        #_zsety = zset[np.random.randint(0,len(zset))]
        #_zsetz = zset[np.random.randint(0,len(zset))]
	node_idx = np.random.randint(0,len(nnodes))
        if (ppn*nnodes[node_idx]==1):
            px=1
            py=1
            pz=1
        elif (ppn*nnodes[node_idx]==2):
            px=2
            py=1
            pz=1
        elif (ppn*nnodes[node_idx]==4):
            px=2
            py=2
            pz=1
        elif (ppn*nnodes[node_idx]==8):
            px=2
            py=2
            pz=2
        elif (ppn*nnodes[node_idx]==16):
            px=4
            py=2
            pz=2
        elif (ppn*nnodes[node_idx]==32):
            px=4
            py=4
            pz=2
        elif (ppn*nnodes[node_idx]==64):
            px=4
            py=4
            pz=4
        elif (ppn*nnodes[node_idx]==128):
            px=8
            py=4
            pz=4
        elif (ppn*nnodes[node_idx]==256):
            px=8
            py=8
            pz=4
        elif (ppn*nnodes[node_idx]==512):
            px=8
            py=8
            pz=8
        elif (ppn*nnodes[node_idx]==1024):
            px=16
            py=8
            pz=8
        elif (ppn*nnodes[node_idx]==2048):
            px=16
            py=16
            pz=8
        elif (ppn*nnodes[node_idx]==4096):
            px=16
            py=16
            pz=16
        elif (ppn*nnodes[node_idx]==8192):
            px=32
            py=16
            pz=16
        elif (ppn*nnodes[node_idx]==16384):
            px=32
            py=32
            pz=16
        elif (ppn*nnodes[node_idx]==32768):
            px=32
            py=32
            pz=32
        else:
            print(ppn,nnodes[node_idx])
            assert(0)
        _zsetx = 1#px
        _zsety = 1#py
        _zsetz = 1#pz
        if (_dset*_zsetx > mode_range_max[3] and _dset*_zsety > mode_range_max[3] and _dset*_zsetz > mode_range_max[3]):
            continue
        is_legal = False
        inner_iter = 0
        while (inner_iter<1000):
            _groups = max(8,_gset)*generate_sample(1,mode_range_max[0]/max(8,_gset)+1,1,1,2)
            _quad = _dset*generate_sample(1,mode_range_max[2]/_dset+1,1,1,2)
            #_zonex = _dset*_zsetx*generate_sample(1,mode_range_max[3]/(_dset*_zsetx)+1,1,1,2)
            #_zoney = _zonex
            #_zonez = _zonex
            inner_iter += 1
            if (_groups > mode_range_max[0] or _groups % _gset != 0):
                continue
            if (_quad > mode_range_max[2] or _quad % _dset != 0):
                continue
            #if (_zonex % (_dset*_zsetx) != 0 or _zoney % (_dset*_zsety) != 0 or _zonez % (_dset*_zsetz) != 0):
            #    continue
            #if (_zonex > mode_range_max[3] or _zoney > mode_range_max[3] or _zonez > mode_range_max[3]):
            #    continue
            #if (_zonex % px != 0 or _zoney % py != 0 or _zonez % pz != 0):
            #    continue
            #if (_zonex*_zoney*_zonez > 16384):
            #    continue
            is_legal = True
            break
        if (is_legal == False):
            continue
        #print(ppn,nnodes[node_idx])
        config = (_groups,_legendre,_quad,_zonex,_zoney,_zonez,_layout,px,py,pz,_dset,_gset,_zsetx,_zsety,_zsetz,_solve)
        print(j,config)
        if (config in config_dict):
            print("REPEAT")
            config_dict[config] += 1
        else:
            config_dict[config] = 1
	write_files[file_dict[(nnodes[node_idx],ppn,thread)]].write("ibrun -n %d /work2/05608/tg849075/kripke_libs/Kripke/build_MPI_OpenMP/bin/kripke.exe --arch OpenMP --groups %d --legendre %d --quad %d --zones %d,%d,%d --layout %s --procs %d,%d,%d --dset %d --gset %d --zset %d,%d,%d --niter 3 --pmethod %s\n"%(ppn*nnodes[node_idx],_groups,_legendre,_quad,_zonex,_zoney,_zonez,_layout,px,py,pz,_dset,_gset,_zsetx,_zsety,_zsetz,_solve))
        if (_groups not in group_dict):
            group_dict[_groups] = 1
        else:
            group_dict[_groups] += 1
        if (_quad not in quad_dict):
            quad_dict[_quad] = 1
        else:
            quad_dict[_quad] += 1
        j += 1
    print("group_dict - ", group_dict)
    print("quad_dict - ", quad_dict)
    print("config_dict - ", config_dict)
