import ctf, time, copy, argparse
import numpy as np
import numpy.linalg as la
import scipy.stats as scst
import random as rand
import pandas as pd
import arg_defs as arg_defs

import backend.ctf_ext as tenpy
from backend.cpd_opt import cpd_als,cpd_amn
from util import extract_datasets

glob_comm = ctf.comm()

def get_midpoint(idx, _nodes, spacing_id):
    # idx is assumed to be the leading coordinate of a cell. Therefore, idx+1 is always valid
    if (spacing_id == 0):
        mid = int(_nodes[idx]*1. + (_nodes[idx+1]-_nodes[idx])/2.)
    elif (spacing_id == 1):
        scale = _nodes[-1]*1./_nodes[-2]
        mid = int(scale**(np.log(_nodes[idx])/np.log(scale) + ((np.log(_nodes[idx+1])/np.log(scale))-(np.log(_nodes[idx])/np.log(scale)))/2.))
    else:
        assert(0)
    return mid

def get_cell_index(val, _nodes):
    if (val >= _nodes[len(_nodes)-1]):
        return len(_nodes)-2
    # Binary Search
    # Loop invariant : cell index is in [start,end]
    start=0
    end=len(_nodes)-2
    save_cell_index = -1
    while (start <= end):
        mid = start + (end-start)/2
        if (val >= _nodes[mid] and val < _nodes[mid+1]):
            return mid
        elif (val <= _nodes[mid]):
            end = mid
        elif (val > _nodes[mid]):
            start = mid+1
        else:
            assert(0)
    return start

def get_node_index(val, _nodes, spacing_id):
    if (val >= _nodes[len(_nodes)-1]):
        return len(_nodes)-1
    if (val <= _nodes[0]):
        return 0
    # Binary Search
    # Loop invariant : cell index is in [start,end]
    start=0
    end=len(_nodes)-2
    save_node_index = -1
    while (start <= end):
        mid = start + (end-start)/2
        if (val >= _nodes[mid] and val < _nodes[mid+1]):
            if (val <= get_midpoint(mid,_nodes,spacing_id)):
                return mid
            else:
                return mid+1
        elif (val <= _nodes[mid]):
            end = mid
        elif (val > _nodes[mid]):
            start = mid+1
        else:
            assert(0)
    return start

def generate_nodes(_min,_max,num_grid_pts,spacing_type):
    cell_nodes = []
    if (spacing_type == 0):
        cell_nodes = np.linspace(_min,_max,num_grid_pts)
    elif (spacing_type == 1):
	cell_nodes = np.geomspace(_min,_max,num_grid_pts)
        cell_nodes[0]=_min
        cell_nodes[-1]=_max
    return cell_nodes

class cpr_model():
    def __init__(self,cp_rank,cp_rank_for_extrapolation,loss_function,reg,max_spline_degree,interpolation_map,response_transform,\
                 sweep_tol,max_num_sweeps,tol_newton,max_num_newton_iter,\
                 barrier_start,barrier_stop,barrier_reduction_factor,cell_spacing,\
                 ngrid_pts,mode_range_min,mode_range_max,build_extrapolation_model):
        self.cp_rank = cp_rank
        self.cp_rank_for_extrapolation = cp_rank_for_extrapolation
        self.loss_function = loss_function
        self.reg = reg
        self.max_spline_degree = max_spline_degree
        self.response_transform = response_transform
        self.sweep_tol = sweep_tol
        self.max_num_sweeps = max_num_sweeps
        self.tol_newton = tol_newton
        self.max_num_newton_iter = max_num_newton_iter
        self.barrier_start = barrier_start
        self.barrier_stop = barrier_stop
        self.barrier_reduction_factor = barrier_reduction_factor
        self.cell_spacing = cell_spacing
	self.cell_nodes = []
	self.contract_str = ''
        self.num_grid_pts = 1
	self.FM1 = []
	self.FM2 = []
	self.extrap_params = []
        self.build_extrapolation_model = build_extrapolation_model

        dim = len(cell_spacing)
	for i in range(dim):
	    self.cell_nodes.append(np.rint(generate_nodes(mode_range_min[i],mode_range_max[i],ngrid_pts[i],cell_spacing[i])))
            self.num_grid_pts *= len(self.cell_nodes[-1])
	    self.contract_str += 'r,'
	    for j in range(1,len(self.cell_nodes[-1])):
		if (self.cell_nodes[-1][j] <= self.cell_nodes[-1][j-1]):
		    self.cell_nodes[-1][j] = self.cell_nodes[-1][j-1] + 1
	self.contract_str = self.contract_str[:-1]
	print("cell_nodes: ", self.cell_nodes)
	#print("contract_str: ", self.contract_str)

        """
        NOTE: No benefit in unfolding tensor along certain modes. It just makes the tensor larger
              and thus requiring a larger training set to achieve suficiently larger projection sets.
	tensor_map = range(len(cell_nodes))
	if (args.tensor_map != ''):
	    tensor_map = [int(n) for n in args.tensor_map.split(',')]
        """
	self.tensor_mode_lengths = [1]*len(self.cell_nodes)
	for i in range(len(self.tensor_mode_lengths)):
	    #tensor_mode_lengths[tensor_map[i]] = len(self.cell_nodes[i])
	    self.tensor_mode_lengths[i] = len(self.cell_nodes[i])
	#print("tensor mode lengths: ", self.tensor_mode_lengths)

	self.interp_map = [1]*len(self.cell_nodes)
	if (interpolation_map != ''):
	    self.interp_map = [int(n) for n in interpolation_map.split(',')]
	self.interp_modes = []
	for i in range(len(self.interp_map)):
	    if self.interp_map[i]==1:
		self.interp_modes.append(i)

    def fit(self,inputs,data):
	omega = ctf.tensor(tuple(self.tensor_mode_lengths), sp=True)
	Tsparse = ctf.tensor(tuple(self.tensor_mode_lengths), sp=True)

	# Use dictionaries to save the sizes of Omega_i
	Projected_Omegas = [[] for i in range(len(self.tensor_mode_lengths))]
	for i in range(len(self.tensor_mode_lengths)):
	    for j in range(len(self.cell_nodes[i])):
		Projected_Omegas[i].append(0)

	start_time = time.time()
	nodes = []
	density = 0.
	training_node_list = []
	training_data_list = []
	save_training_nodes = []
	node_data_dict = {}
	node_count_dict = {}
	for i in range(len(data)):
	    input_tuple = inputs[i,:]
	    node_key = []
	    for j in range(len(input_tuple)):
		node_key.append(get_node_index(input_tuple[j],self.cell_nodes[j],self.cell_spacing[j]))
	    save_training_nodes.append(node_key)
	    if (tuple(node_key) not in node_data_dict):
		node_count_dict[tuple(node_key)] = 1
		node_data_dict[tuple(node_key)] = data[i]
	    else:
		node_count_dict[tuple(node_key)] += 1
		node_data_dict[tuple(node_key)] += data[i]
	    for j in range(len(input_tuple)):
		Projected_Omegas[j][node_key[j]] += 1
	density = len(node_data_dict.keys())*1./self.num_grid_pts
	for key in node_count_dict.keys():
	    training_node_list.append(key)
	    training_data_list.append(node_data_dict[key]/node_count_dict[key])
	node_data = np.array(training_data_list)

	#print("Projected_Omegas: ",Projected_Omegas)
        #print("Density - %f"%(density))

	omega.write(training_node_list,np.ones(len(training_node_list)))
	Tsparse.write(training_node_list,node_data.reshape(len(training_node_list)))

	FM1 = []
	FM2 = []
	for k in range(len(self.tensor_mode_lengths)):
	    FM1.append(ctf.tensor((Tsparse.shape[k],self.cp_rank)))
	    FM2.append(ctf.tensor((Tsparse.shape[k],self.cp_rank_for_extrapolation)))
	    FM1[-1].fill_random()
	    FM2[-1].fill_random(0,.01)
	# Optimize model
	# For interpolation, we first minimize mean squared error using log-transformed data
	_T_ = Tsparse.copy()
	if (self.response_transform==0):
	    pass
	elif (self.response_transform==1):
	    [inds,data] = _T_.read_local_nnz()
	    data = np.log(data)
	    _T_.write(inds,data)
	else:
	    assert(0)
	FM1,loss1,num_sweeps1 = cpd_als("MSE", tenpy, _T_,
				     omega, FM1, self.reg, self.sweep_tol, self.max_num_sweeps)
	num_newton_iter1 = 0
        for k in range(len(self.tensor_mode_lengths)):
	    self.FM1.append(FM1[k].to_nparray())
	# Only need to attain extrapolation model if extrapolation is relevant.
	if (len(self.interp_modes)>0 and self.build_extrapolation_model==1):
	    # For extrapolation, we minimize MLogQ2
	    _T_ = Tsparse.copy()
	    FM2,loss2,num_sweeps2,num_newton_iter2 =  cpd_amn("MLogQ2",\
	      tenpy, _T_, omega, FM2, self.reg,\
	      self.sweep_tol,self.max_num_sweeps, self.tol_newton,\
	      self.max_num_newton_iter, self.barrier_start,self.barrier_stop,self.barrier_reduction_factor)
            for k in range(len(self.tensor_mode_lengths)):
	        self.FM2.append(FM2[k].to_nparray())
                """
                # Addition for now, remove later. Note only works for rank-1 because only 1 column
                scale = la.norm(self.FM2[-1][:,0],2)
	        self.FM2[-1] /= scale
                """
            """
	    for i in range(len(self.interp_modes)):	# One cannot simply extrapolate only certain modes.
		self.extrap_params.append([])
		#NOTE: Could try 3 instead of 2 to try quadratic global models
		ls_mat = np.ones(shape=(len(self.FM2[self.interp_modes[i]][:,0]),1+self.max_spline_degree))
		#NOTE: Below might need to be log-transformed?
		for j in range(self.max_spline_degree):
		    ls_mat[:,1+j] = self.cell_nodes[self.interp_modes[i]]**(1+j)
		for j in range(len(self.FM2[self.interp_modes[i]][0,:])):
		    #NOTE: least-squares regression with no data transformation is
		    #      permissable here.
		    lsq_params,ret2,_,_ = la.lstsq(ls_mat[:,:],self.FM2[self.interp_modes[i]][:,j])
		    self.extrap_params[-1].append(lsq_params)
            """
	    for i in range(len(self.tensor_mode_lengths)):	# One cannot simply extrapolate only certain modes.
		self.extrap_params.append([])
		#NOTE: Could try 3 instead of 2 to try quadratic global models
		ls_mat = np.ones(shape=(len(self.FM2[i][:,0]),1+self.max_spline_degree))
		#NOTE: Below might need to be log-transformed?
		for j in range(self.max_spline_degree):
		    ls_mat[:,1+j] = self.cell_nodes[i]**(1+j)
		for j in range(len(self.FM2[i][0,:])):
		    #NOTE: least-squares regression with no data transformation is
		    #      permissable here.
		    lsq_params,ret2,_,_ = la.lstsq(ls_mat[:,:],self.FM2[i][:,j])
		    self.extrap_params[-1].append(lsq_params)
            if (self.loss_function == 1):
                self.FM1 = self.FM2	# Copy here so that in predict(...), the FM2 factor matrices are used.
        else:
            loss2 = 0
        return (self.num_grid_pts,density,loss1,loss2)

    def predict(self,input_tuple):
        node = []
        for j in range(len(input_tuple)):
            node.append(get_node_index(input_tuple[j],self.cell_nodes[j],self.cell_spacing[j]))
	midpoints = []
	local_interp_modes = []
	local_interp_map = [0]*len(input_tuple)
	decisions = [0]*len(input_tuple)
	is_interpolation = True
	for j in range(len(self.interp_modes)):
	    #cell_node_idx = interp_modes[j]
	    # check if input_tuple[interp_modes[j]] is outside of the cell_nodes on either side
	    left_midpoint = get_midpoint(0, self.cell_nodes[self.interp_modes[j]], self.cell_spacing[self.interp_modes[j]])
	    right_midpoint = get_midpoint(len(self.cell_nodes[self.interp_modes[j]])-2, self.cell_nodes[self.interp_modes[j]], self.cell_spacing[self.interp_modes[j]])
	    if (input_tuple[self.interp_modes[j]] < self.cell_nodes[self.interp_modes[j]][0] and self.build_extrapolation_model == 1):
		# extrapolation necessary: outside range of bounding box on left
		decisions[self.interp_modes[j]]=3
		is_interpolation = False
	    elif (input_tuple[self.interp_modes[j]] > self.cell_nodes[self.interp_modes[j]][-1] and self.build_extrapolation_model == 1):
		# extrapolation necessary: outside range of bounding box on right
		decisions[self.interp_modes[j]]=4
		is_interpolation = False
	    elif (input_tuple[self.interp_modes[j]] < left_midpoint):
		# extrapolation necessary: inside range of bounding box on left, but left of left-most midpoint
		decisions[self.interp_modes[j]]=1
	    elif (input_tuple[self.interp_modes[j]] > right_midpoint):
		# extrapolation necessary: inside range of bounding box on right, but right of right-most midpoint
		decisions[self.interp_modes[j]]=2
	    else:
		midpoints.append(get_midpoint(get_cell_index(input_tuple[self.interp_modes[j]],self.cell_nodes[self.interp_modes[j]]), self.cell_nodes[self.interp_modes[j]], self.cell_spacing[self.interp_modes[j]]))
		local_interp_modes.append(self.interp_modes[j])
		local_interp_map[self.interp_modes[j]] = 1
		decisions[self.interp_modes[j]]=5
	element_index_modes_list = []
	if (is_interpolation == True or self.build_extrapolation_model==0):
	    for j in range(len(local_interp_modes)):
		element_index_modes_list.append([])
		for xx in range(2):
		    if (input_tuple[local_interp_modes[j]] <= midpoints[j]):
			element_index_modes_list[-1].append(node[local_interp_modes[j]]-(2-1)/2+xx)
		    else:
			element_index_modes_list[-1].append(node[local_interp_modes[j]]-2/2+xx)
	    model_val = 0.
	    # Do not consider extrapolation modes
	    for j in range(2**len(local_interp_modes)):
		interp_id = j
		interp_id_list = [0]*len(local_interp_modes)
		counter = 0
		while (interp_id>0):
		    interp_id_list[counter] = interp_id%2
		    interp_id /= 2
		    counter += 1
		coeff = 1
		for l in range(len(local_interp_modes)):
		    cell_node_idx = local_interp_modes[l]
		    for ll in range(2):
			if (ll != interp_id_list[l]):
			    coeff *= (input_tuple[local_interp_modes[l]]-self.cell_nodes[cell_node_idx][element_index_modes_list[l][ll]])\
				     /(self.cell_nodes[cell_node_idx][element_index_modes_list[l][interp_id_list[l]]]-self.cell_nodes[cell_node_idx][element_index_modes_list[l][ll]])
		factor_row_list = []
		interp_counter = 0
		for l in range(len(input_tuple)):
		    if (local_interp_map[l]==1):
			factor_row_list.append(self.FM1[l][element_index_modes_list[interp_counter][interp_id_list[interp_counter]],:])
			interp_counter += 1
		    else:
			if (decisions[l]==0):	# categorical or non-numerical parameter in which interpolation/extrapolation is not relevant
			    factor_row_list.append(self.FM1[l][node[l],:])
			elif (decisions[l]==1):
			    row_data = []
			    for ll in range(len(self.FM1[l][0,:])):
				row_data.append(self.FM1[l][0,ll] + (input_tuple[l]-self.cell_nodes[l][0])/(self.cell_nodes[l][1]-self.cell_nodes[l][0])*(self.FM1[l][1,ll]-self.FM1[l][0,ll]))
			    factor_row_list.append(np.array(row_data))
			elif (decisions[l]==2):
			    row_data = []
			    for ll in range(len(self.FM1[l][0,:])):
				row_data.append(self.FM1[l][-2,ll] + (input_tuple[l]-self.cell_nodes[l][-2])/(self.cell_nodes[l][-1]-self.cell_nodes[l][-2])*(self.FM1[l][-1,ll]-self.FM1[l][-2,ll]))
			    factor_row_list.append(np.array(row_data))
		t_val = np.einsum(self.contract_str,*factor_row_list)
		if (self.response_transform==0):
		    pass
		elif (self.response_transform==1):
		    t_val = np.exp(1)**t_val
		    pass
		else:
		    assert(0)
		model_val += coeff * t_val
	    return model_val
	else:
	    # Rank-k prediction
            model_prediction = 0
            for lll in range(self.cp_rank_for_extrapolation):
		model_val = 1.
		for l in range(len(input_tuple)):
                    #TODO: If we don't need to extrapolate along this mode, we can use the interpolation scheme above. So do this on a per-mode basis rather than all-or-nothing
                    if (self.interp_map[l]==1):
			factor_matrix_contribution = 0
			for ll in range(1+self.max_spline_degree):
			    #TODO: Use horner's rule for faster eval (not needed for small self.max_spline_degree)
			    factor_matrix_contribution += self.extrap_params[l][lll][ll]*(input_tuple[l]**ll)
                    else:
                        factor_matrix_contribution = self.FM2[l][node[l],lll]
		    model_val *= factor_matrix_contribution
                model_prediction += model_val
	    return model_prediction

    def print_parameters(self):
        print("Extrapolation model")
        print(self.extrap_params)
        normalization_factors = [1]*len(self.FM1[0][0,:])
        for i in range(len(self.FM1[0][0,:])):
            for j in range(len(self.FM1)):
                scale = la.norm(self.FM1[j][:,i],2)
                normalization_factors[i] *= scale
                #self.FM1[j][:,i] /= scale
        # Print normalization constants
        print("Normalization scaling factors")
        for i in range(len(normalization_factors)):
            print("%f,"%(normalization_factors[i])),
        print("")
        normalization_factors = [1]*len(self.FM2[0][0,:])
        for i in range(len(self.FM2[0][0,:])):
            for j in range(len(self.FM2)):
                scale = la.norm(self.FM2[j][:,i],2)
                normalization_factors[i] *= scale
                #self.FM2[j][:,i] /= scale
        # Print normalization constants
        print("Normalization scaling factors")
        for i in range(len(normalization_factors)):
            print("%f,"%(normalization_factors[i])),
        print("")
        # Print factor matrices
        for i in range(len(self.FM1)):
            print("Factor matrix %i"%(i))
            for k in range(len(self.FM1[i][:,0])):
                print("%f,"%(self.cell_nodes[i][k])),
                for j in range(len(self.FM1[i][0,:])):
                    #val = self.FM1[i][k,j]
                    scale = la.norm(self.FM1[i][:,j],2)
                    #print("%f,"%(self.FM1[i][k,j]/(normalization_factors[j]**(1./3.)))),
                    print("%f,"%(self.FM1[i][k,j]/scale)),
                print("")
        # Print factor matrices
        for i in range(len(self.FM2)):
            print("Factor matrix %i"%(i))
            for k in range(len(self.FM2[i][:,0])):
                print("%f,"%(self.cell_nodes[i][k])),
                for j in range(len(self.FM2[i][0,:])):
                    #val = self.FM2[i][k,j]
                    scale = la.norm(self.FM2[i][:,j],2)
                    print("%f,"%(self.FM2[i][k,j]))/scale)),
                print("")
