import ctf, time, copy, argparse
import numpy as np
import numpy.linalg as la
import scipy.stats as scst
import random as rand
import pandas as pd
import arg_defs as arg_defs

import pyearth as pe	#NOTE: Invalid for Python3

import backend.ctf_ext as tenpy
from backend.cpd_opt import cpd_als,cpd_amn
from util import extract_datasets

glob_comm = ctf.comm()

def get_midpoint(idx, _nodes, spacing_id):
    # idx is assumed to be the leading coordinate of a cell. Therefore, idx+1 is always valid
    if (spacing_id == 0):
        mid = int(_nodes[idx]*1. + (_nodes[idx+1]-_nodes[idx])/2.)
    elif (spacing_id == 1 or spacing_id == 2):
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
        mid = start + (end-start)//2
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

def generate_nodes(_min,_max,num_grid_pts,spacing_type,custom_grid_pts=[]):
    cell_nodes = []
    if (spacing_type == 0):
        cell_nodes = np.linspace(_min,_max,num_grid_pts)
    elif (spacing_type == 1):
        cell_nodes = np.geomspace(_min,_max,num_grid_pts)
        cell_nodes[0]=_min
        cell_nodes[-1]=_max
    elif (spacing_type == 2):
        cell_nodes = np.array(custom_grid_pts)
        cell_nodes[0]=_min
        cell_nodes[-1]=_max
    return cell_nodes

class cpr_model():
    def __init__(self,ngrid_pts,interpolation_map,cell_spacing,mode_range_min,mode_range_max,\
                 cp_rank=3,cp_rank_for_extrapolation=1,loss_function=0,reg=1e-5,max_spline_degree=3,\
                 response_transform=1,custom_grid_pts=[],sweep_tol=1e-5,max_num_sweeps=50,tol_newton=1e-3,max_num_newton_iter=40,\
                 barrier_start=1e1,barrier_stop=1e-11,barrier_reduction_factor=8,projection_set_size_threshold_=[],build_extrapolation_model=1,save_dataset=False):

        assert(len(ngrid_pts) == len(interpolation_map))
        assert(len(ngrid_pts) == len(cell_spacing))
        assert(len(ngrid_pts) == len(mode_range_min))
        assert(len(ngrid_pts) == len(mode_range_max))

        self.cp_rank = cp_rank
        self.cp_rank_for_extrapolation = cp_rank_for_extrapolation
        self.loss_function = loss_function
        self.reg = reg
        self.max_spline_degree = max_spline_degree
        self.response_transform = response_transform
        self.custom_grid_pts = list(custom_grid_pts)
        self.sweep_tol = sweep_tol
        self.max_num_sweeps = max_num_sweeps
        self.tol_newton = tol_newton
        self.max_num_newton_iter = max_num_newton_iter
        self.barrier_start = barrier_start
        self.barrier_stop = barrier_stop
        self.barrier_reduction_factor = barrier_reduction_factor
        self.cell_spacing = list(cell_spacing)
        self.cell_nodes = []
        self.contract_str = ''
        self.num_grid_pts = 1
        self.FM1 = []
        self.FM2 = []
        self.FM2_sv = []
        self.extrap_models = []
        self.build_extrapolation_model = build_extrapolation_model
        self.save_dataset = save_dataset
        self.yi = []
        self.Xi = []

        dim = len(cell_spacing)
        start_grid_idx = 0
        for i in range(dim):
            if (cell_spacing[i] != 2):
                self.cell_nodes.append(np.rint(generate_nodes(mode_range_min[i],mode_range_max[i],ngrid_pts[i],cell_spacing[i])))
            else:
                self.cell_nodes.append(np.rint(generate_nodes(mode_range_min[i],mode_range_max[i],ngrid_pts[i],cell_spacing[i],self.custom_grid_pts[start_grid_idx:start_grid_idx+ngrid_pts[i]])))
                start_grid_idx += ngrid_pts[i]
            self.contract_str += 'r,'
            for j in range(1,len(self.cell_nodes[-1])):
                if (self.cell_nodes[-1][j] <= self.cell_nodes[-1][j-1]):
                    self.cell_nodes[-1][j] = self.cell_nodes[-1][j-1] + 1
            self.num_grid_pts *= len(self.cell_nodes[-1])
        self.contract_str = self.contract_str[:-1]
        #print("cell_nodes: ", self.cell_nodes)
        #print("contract_str: ", self.contract_str)

        """
        NOTE: No benefit in unfolding tensor along certain modes. It just makes the tensor larger
              and thus requiring a larger training set to achieve sufficiently larger projection sets.
        tensor_map = range(len(cell_nodes))
        if (args.tensor_map != ''):
            tensor_map = [int(n) for n in args.tensor_map.split(',')]
        """
        self.tensor_mode_lengths = [1]*len(self.cell_nodes)
        for i in range(len(self.tensor_mode_lengths)):
            self.tensor_mode_lengths[i] = len(self.cell_nodes[i])
        self.Projected_Omegas = [[] for i in range(len(self.tensor_mode_lengths))]

        self.interp_map = list(interpolation_map)
        self.projection_set_size_threshold = [8]*len(self.cell_nodes)
        if (projection_set_size_threshold_ != []):
            self.projection_set_size_threshold = list(projection_set_size_threshold_)
        self.numerical_modes = []
        self.ordinal_modes = []
        self.categorical_modes = []
        for i in range(len(self.interp_map)):
            if self.interp_map[i]==1:
                self.numerical_modes.append(i)
            elif self.interp_map[i]==2:
                self.ordinal_modes.append(i)
            else:
                self.categorical_modes.append(i)

    def fit(self,inputs,data):
        if (self.save_dataset):
            self.Xi = inputs.copy()
            self.yi = data.copy()
        omega = ctf.tensor(tuple(self.tensor_mode_lengths), sp=True)
        Tsparse = ctf.tensor(tuple(self.tensor_mode_lengths), sp=True)

        # Use dictionaries to save the sizes of Omega_i
        for i in range(len(self.tensor_mode_lengths)):
            for j in range(len(self.cell_nodes[i])):
                self.Projected_Omegas[i].append(0)

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
                self.Projected_Omegas[j][node_key[j]] += 1
        density = len(node_data_dict.keys())*1./self.num_grid_pts
        for key in node_count_dict.keys():
            training_node_list.append(key)
            training_data_list.append(node_data_dict[key]/node_count_dict[key])
        node_data = np.array(training_data_list)

        print("Projected_Omegas: ",self.Projected_Omegas)
        print("Density - %f"%(density))

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
        #   NOTE: above statement is no longer true with recent optimizations
        if (len(self.numerical_modes)>0 or len(self.ordinal_modes)>0 or self.build_extrapolation_model==1):
            # For extrapolation, we minimize MLogQ2
            _T_ = Tsparse.copy()
            # NOTE: I changed self.max_num_sweeps to 5 here, which is often sufficient
            FM2,loss2,num_sweeps2,num_newton_iter2 =  cpd_amn("MLogQ2",\
              tenpy, _T_, omega, FM2, self.reg,\
              self.sweep_tol,5, self.tol_newton,\
              self.max_num_newton_iter, self.barrier_start,self.barrier_stop,self.barrier_reduction_factor)
            for k in range(len(self.tensor_mode_lengths)):
                self.FM2.append(FM2[k].to_nparray())
                """
                # Addition for now, remove later. Note only works for rank-1 because only 1 column
                scale = la.norm(self.FM2[-1][:,0],2)
                self.FM2[-1] /= scale
                """
            updated_cell_nodes = []
            for i in range(len(self.FM2)):
                updated_cell_nodes.append([])
                if (self.interp_map[i] == 0):
                    self.FM2_sv.append(())
                    continue
                # Inspect the FM to identify the projected set size
                local_projected_set_size = 0
                for j in range(self.FM2[i].shape[0]):
                    if (self.Projected_Omegas[i][j] >= min(self.projection_set_size_threshold[i],max(self.Projected_Omegas[i]))):
                        local_projected_set_size = local_projected_set_size + 1
                        updated_cell_nodes[-1].append(self.cell_nodes[i][j])
                Factor_Matrix__ = np.zeros(shape=(local_projected_set_size,self.FM2[i].shape[1]))
                column_count = 0
                for j in range(self.FM2[i].shape[0]):
                    if (self.Projected_Omegas[i][j] >= min(self.projection_set_size_threshold[i],max(self.Projected_Omegas[i]))):
                        for k in range(self.FM2[i].shape[1]):
                            Factor_Matrix__[column_count,k] = self.FM2[i][j,k]
                        column_count = column_count + 1
                #U,S,VT = la.svd(self.FM2[i])
                #print(i,updated_cell_nodes[i],self.Projected_Omegas[i],self.FM2[i],Factor_Matrix__)
                U,S,VT = la.svd(Factor_Matrix__)
                self.FM2_sv.append((U[:,0],S[0],VT[0,:]))
                # Identify Perron vector: simply flip signs if necessary
                for j in range(len(self.FM2_sv[-1][0])):
                    if (self.FM2_sv[-1][0][j]<0):
                        self.FM2_sv[-1][0][j] *= (-1)
                for j in range(len(self.FM2_sv[-1][2])):
                    if (self.FM2_sv[-1][2][j]<0):
                        self.FM2_sv[-1][2][j] *= (-1)
            for i in range(len(self.tensor_mode_lengths)):        # One cannot simply extrapolate only certain modes.
                if (self.interp_map[i] == 0):
                    self.extrap_models.append([])
                    continue
                self.extrap_models.append(pe.Earth(max_terms=10,max_degree=self.max_spline_degree,allow_linear=True))
                #self.extrap_models[-1].fit(np.log(self.cell_nodes[i].reshape(-1,1)),np.log(self.FM2_sv[i][0]))
                print(np.array(updated_cell_nodes[i]).reshape(-1,1),self.FM2_sv[i][0])
                self.extrap_models[-1].fit(np.log(np.array(updated_cell_nodes[i]).reshape(-1,1)),np.log(self.FM2_sv[i][0]))
            if (self.loss_function == 1):
                self.FM1 = self.FM2        # Copy here so that in predict(...), the FM2 factor matrices are used.
        else:
            loss2 = 0

        return (self.num_grid_pts,density,loss1,loss2)

    def predict(self,input_tuple):
        node = []
        for j in range(len(input_tuple)):
            node.append(get_node_index(input_tuple[j],self.cell_nodes[j],self.cell_spacing[j]))
        midpoints = []
        local_numerical_modes = []
        local_interp_map = [0]*len(input_tuple)
        decisions = [0]*len(input_tuple)
        is_interpolation = True
        for j in range(len(self.numerical_modes)):
            #cell_node_idx = numerical_modes[j]
            # check if input_tuple[numerical_modes[j]] is outside of the cell_nodes on either side
            left_midpoint = get_midpoint(0, self.cell_nodes[self.numerical_modes[j]], self.cell_spacing[self.numerical_modes[j]])
            right_midpoint = get_midpoint(len(self.cell_nodes[self.numerical_modes[j]])-2, self.cell_nodes[self.numerical_modes[j]], self.cell_spacing[self.numerical_modes[j]])
            if (input_tuple[self.numerical_modes[j]] < self.cell_nodes[self.numerical_modes[j]][0] and self.build_extrapolation_model == 1):
                # extrapolation necessary: outside range of bounding box on left
                decisions[self.numerical_modes[j]]=3
                is_interpolation = False
            elif (input_tuple[self.numerical_modes[j]] > self.cell_nodes[self.numerical_modes[j]][-1] and self.build_extrapolation_model == 1):
                # extrapolation necessary: outside range of bounding box on right
                decisions[self.numerical_modes[j]]=4
                is_interpolation = False
            elif (input_tuple[self.numerical_modes[j]] < left_midpoint):
                # extrapolation necessary: inside range of bounding box on left, but left of left-most midpoint
                decisions[self.numerical_modes[j]]=1
            elif (input_tuple[self.numerical_modes[j]] > right_midpoint):
                # extrapolation necessary: inside range of bounding box on right, but right of right-most midpoint
                decisions[self.numerical_modes[j]]=2
            else:
                midpoints.append(get_midpoint(get_cell_index(input_tuple[self.numerical_modes[j]],self.cell_nodes[self.numerical_modes[j]]), self.cell_nodes[self.numerical_modes[j]], self.cell_spacing[self.numerical_modes[j]]))
                local_numerical_modes.append(self.numerical_modes[j])
                local_interp_map[self.numerical_modes[j]] = 1
                decisions[self.numerical_modes[j]]=5
        ordinal_prediction_switch = []
        for j in range(len(self.ordinal_modes)):
            # check if any of the corresponding projection sets is too small
            if (self.Projected_Omegas[self.ordinal_modes[j]][node[self.ordinal_modes[j]]] < min(self.projection_set_size_threshold[self.ordinal_modes[j]],max(self.Projected_Omegas[self.ordinal_modes[j]]))):
                is_interpolation = False
                ordinal_prediction_switch.append(1)
            else:
                ordinal_prediction_switch.append(0)
        element_index_modes_list = []
        for j in range(len(local_numerical_modes)):
            element_index_modes_list.append([])
            for xx in range(2):
                if (input_tuple[local_numerical_modes[j]] <= midpoints[j]):
                    element_index_modes_list[-1].append(node[local_numerical_modes[j]]-(2-1)/2+xx)
                else:
                    element_index_modes_list[-1].append(node[local_numerical_modes[j]]-2/2+xx)
        if (is_interpolation == True or self.build_extrapolation_model==0):
            model_val = 0.
            # Do not consider extrapolation modes
            for j in range(2**len(local_numerical_modes)):
                interp_id = j
                interp_id_list = [0]*len(local_numerical_modes)
                counter = 0
                while (interp_id>0):
                    interp_id_list[counter] = interp_id%2
                    interp_id /= 2
                    counter += 1
                coeff = 1
                for l in range(len(local_numerical_modes)):
                    cell_node_idx = local_numerical_modes[l]
                    for ll in range(2):
                        if (ll != interp_id_list[l]):
                            coeff *= (input_tuple[local_numerical_modes[l]]-self.cell_nodes[cell_node_idx][element_index_modes_list[l][ll]])\
                                     /(self.cell_nodes[cell_node_idx][element_index_modes_list[l][interp_id_list[l]]]-self.cell_nodes[cell_node_idx][element_index_modes_list[l][ll]])
                factor_row_list = []
                interp_counter = 0
                for l in range(len(input_tuple)):
                    if (local_interp_map[l]==1):
                        factor_row_list.append(self.FM1[l][element_index_modes_list[interp_counter][interp_id_list[interp_counter]],:])
                        interp_counter += 1
                    else:
                        if (decisions[l]==0):        # categorical or non-numerical parameter in which interpolation/extrapolation is not relevant
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
            model_prediction = 0
            for lll in range(self.cp_rank_for_extrapolation):
                model_val = 1.
                ordinal_count = 0
                for l in range(len(input_tuple)):
                    if (self.interp_map[l]==1):
                        if (local_interp_map[l]==0):
                            factor_matrix_contribution = np.exp(1)**self.extrap_models[l].predict([np.log(input_tuple[l])])[0]*self.FM2_sv[l][1]*self.FM2_sv[l][2][lll]
                        else:
                            factor_matrix_contribution = self.FM2[l][node[l],lll] + (input_tuple[l]-self.cell_nodes[l][node[l]]) * (self.FM2[l][node[l]+1,lll] - self.FM2[l][node[l],lll])\
                                     /(self.cell_nodes[l][node[l]+1]-self.cell_nodes[l][node[l]])
                    elif (self.interp_map[l]==2):
                        if (ordinal_prediction_switch[ordinal_count]==1):
                            factor_matrix_contribution = np.exp(1)**self.extrap_models[l].predict([np.log(input_tuple[l])])[0]*self.FM2_sv[l][1]*self.FM2_sv[l][2][lll]
                        else:
                            factor_matrix_contribution = self.FM2[l][node[l],lll]
                        ordinal_count = ordinal_count + 1
                    else:
                        factor_matrix_contribution = self.FM2[l][node[l],lll]
                    model_val *= factor_matrix_contribution
                model_prediction += model_val
            return model_prediction

    # NOTE: The exhaustive-search method below is not efficient for high-dimensional configuration spaces.
    def ask_per_mode(self,min_element,level,config,best_configuration):
        if (level == len(self.cell_nodes)):
            t_val = self.predict(config)
            if (t_val < min_element):
                min_element = t_val
                best_configuration = list(config)
            return
        for i in range(len(self.cell_nodes[level])):
            config.append(self.cell_nodes[level][i])
            factor_row_list.append(self.FM2[level][i,:])
            self.ask_per_mode(min_element,level+1,config,factor_row_list,best_configuration)
            factor_row_list.pop()
            config.pop()

    def ask(self,n_points=None, batch_size=20):
        assert(n_points==1)
        """
        # Initial strategy: search across all elements of tensor
        min_runtime_input = self.ask_per_mode(np.inf,0,[],[])
        print("What is this min_runtime_input - ", min_runtime_input)
        return min_runtime_input
        """
        # Next strategy: leverage rank-1 factorization of each factor matrix by scanning the left-singular vector.
        min_runtime_input = []
        for i in range(len(self.FM2)):
            # For each mode, inspect the projection set size
            if (self.interp_map[i]==0):
                # For categorical modes, we assume that there will be a sufficiently-large number of observations
                #   per parameter value, and thus do not need to inspect this mode's projection set sizes.
                min_row_metric = np.inf
                save_min_row = -1
                for j in range(len(self.FM2[i][:,0])):
                    row_metric = scst.gmean(self.FM2[i][j,:])
                    if (row_metric < min_row_metric):
                        min_row_metric = row_metric
                        save_min_row = j
                assert(save_min_row >= 0)
                min_runtime_input.append(self.cell_nodes[i][save_min_row])
            elif (self.interp_map[i]==1):
                # Can likely just use factor matrix, but not sure how to select the best parameter here
                # For single-task autotuning, we don't need to do anything here, but for multi-task autotuning this becomes interesting
                assert(0)
            elif (self.interp_map[i]==2):
                # For ordinal modes, we must verify to check that the size of the corresponding projection set
                # is sufficiently large, as this will influence what method is chosen for prediction.
                min_row_metric = np.inf
                save_min_row = -1
                for j in range(len(self.FM2[i][:,0])):
                    if (self.Projected_Omegas[i][j] >= min(self.projection_set_size_threshold[i],max(self.Projected_Omegas[i]))):
                        row_metric = scst.gmean(self.FM2[i][j,:])
                    else:
                        save_row_array = []
                        for j2 in range(len(self.FM2[i][0,:])):
                            save_row_array.append(np.exp(1)**self.extrap_models[i].predict([np.log(self.cell_nodes[i][j])])[0]*self.FM2_sv[i][1]*self.FM2_sv[i][2][j2])
                        row_metric = scst.gmean(self.FM2[i][j,:])
                    if (row_metric < min_row_metric):
                        min_row_metric = row_metric
                        save_min_row = j
                assert(save_min_row >= 0)
                min_runtime_input.append(self.cell_nodes[i][save_min_row])
        return min_runtime_input 

    def tell(self,configuration,runtime):
        if (self.save_dataset):
            self.Xi += configuration
            self.yi += runtime
            self.fit(self.Xi,self.yi)
        return

    def print_parameters(self):
        if (self.build_extrapolation_model):
            print("Extrapolation model")
            for i in range(len(self.FM2)):
                if (self.interp_map[i]>0):
                    print(self.extrap_models[i].summary())
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
        if (self.build_extrapolation_model):
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
                    scale = la.norm(self.FM1[i][:,j],2)
                    print("%f,"%(self.FM1[i][k,j]/scale)),
                print("")
        # Print factor matrices
        if (self.build_extrapolation_model):
            for i in range(len(self.FM2)):
                print("Factor matrix %i"%(i))
                for k in range(len(self.FM2[i][:,0])):
                    print("%f,"%(self.cell_nodes[i][k])),
                    for j in range(len(self.FM2[i][0,:])):
                        scale = la.norm(self.FM2[i][:,j],2)
                        print("%f,"%(self.FM2[i][k,j]/scale)),
                    print("")
