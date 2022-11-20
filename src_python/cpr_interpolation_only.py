import ctf, time, copy, argparse
import numpy as np
import numpy.linalg as la
import scipy.stats as scst
import random as rand
import pandas as pd
import arg_defs as arg_defs

import backend.ctf_ext as tenpy
from backend.cpd_opt import cpd_als,cpd_amn

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

def generate_models(reg,nals_sweeps,cp_rank,element_len):
    model_list = []
    for i in reg:
        for j in nals_sweeps:
            for k in cp_rank:
                for l in element_len:
                    model_list.append((i,j,k,l))
    return model_list

def cpc(_T,_omega,_guess,_reg_als,_tol_als,_num_iter_als,_error_metric,_tol_newton, _num_iter_newton,_barrier_start,_barrier_reduction_factor):
    Tincomplete = _T.copy()
    if (_error_metric == "MSE"):
        guess,loss,n_newton_iterations,n_newton_restarts = cpd_als(_error_metric, tenpy, Tincomplete, _omega, _guess, _reg_als, _tol_als, _num_iter_als)
    else:
        guess,loss,n_newton_iterations,n_newton_restarts = cpd_amn(_error_metric,\
                                                             tenpy, Tincomplete, _omega, _guess, _reg_als, _tol_als,\
                                                              _num_iter_als, _tol_newton, _num_iter_newton,\
                                                              _barrier_start,_barrier_reduction_factor)
    return (guess,loss,n_newton_iterations,n_newton_restarts)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    arg_defs.add_general_arguments(parser)
    args, _ = parser.parse_known_args()

    if (ctf.comm().np()==1):
        numpy_eval = 1
    else:
        numpy_eval = args.numpy_eval

    np.random.seed(10)
    print("Location of training data: %s"%(args.training_file))
    print("Location of test data: %s"%(args.test_file))
    print("Location of output data: %s"%(args.output_file))
    training_df = pd.read_csv('%s'%(args.training_file), index_col=0, sep=',')
    test_df = pd.read_csv('%s'%(args.test_file), index_col=0, sep=',')
    print("args.input_columns - ", args.input_columns)
    print("args.data_columns - ", args.data_columns)
    param_list = training_df.columns[[int(n) for n in args.input_columns.split(',')]].tolist()
    data_list = training_df.columns[[int(n) for n in args.data_columns.split(',')]].tolist()
    print("param_list: ", param_list)

    reg = [float(n) for n in args.reg.split(',')]
    nals_sweeps = [int(n) for n in args.nals_sweeps.split(',')]
    cp_rank = [int(n) for n in args.cp_rank.split(',')]
    element_len = [int(n) for n in args.element_mode_len.split(',')]
    print("reg - ", reg)
    print("nals_sweeps - ", nals_sweeps)
    print("cp_rank - ", cp_rank)
    print("element_len - ", element_len)
    assert(len(element_len)<=1)
    if (len(element_len)==1):
        assert(element_len[0]==2)

    # Generate list of model types parameterized on hyper-parameters
    model_list = generate_models(reg,nals_sweeps,cp_rank,element_len)
    print("model_list - ", model_list)

    # Note: assumption that training/test input files follow same format
    x_test = np.array(range(len(test_df[param_list].values)))
    np.random.shuffle(x_test)
    x_train = np.array(range(len(training_df[param_list].values)))
    np.random.shuffle(x_train)

    test_inputs = test_df[param_list].values[x_test]
    test_data = test_df[data_list].values.reshape(-1)[x_test]
    test_set_size = min(args.test_set_size,test_inputs.shape[0])
    split_idx = int(args.test_set_split_percentage * test_set_size)
    validation_set_size = split_idx
    test_set_size = test_inputs.shape[0]-split_idx

    training_inputs = training_df[param_list].values[x_train]
    training_data = training_df[data_list].values.reshape(-1)[x_train]
    training_set_size = min(training_inputs.shape[0],args.training_set_size)
    training_inputs = training_inputs[:training_set_size,:]
    training_data = training_data[:training_set_size]

    print("training_inputs - ", training_inputs)
    print("training_data - ", training_data)
    print("test_inputs - ", test_inputs)
    print("test_data - ", test_data)

    #TODO: Not sure what to do about this yet.
    input_scale = 0
    if (args.scale_mode == "weak"):
        mult = param_space_dim

    test_inputs = test_inputs.astype(np.float64)
    training_inputs = training_inputs.astype(np.float64)
    mode_range_min = [0]*len(param_list)
    mode_range_max = [0]*len(param_list)
    if (args.mode_range_min == '' or args.mode_range_max == ''):
	for i in range(training_inputs.shape[1]):
	    mode_range_min[i] = np.amin(training_inputs[:,i])
	    mode_range_max[i] = np.amax(training_inputs[:,i])
    else:
        mode_range_min = [float(n) for n in args.mode_range_min.split(',')]
        mode_range_max = [float(n) for n in args.mode_range_max.split(',')]
    print("str mode_range_min - ", args.mode_range_min)
    print("str mode_range_max - ", args.mode_range_max)
    print("mode_range_min - ", mode_range_min)
    print("mode_range_max - ", mode_range_max)
    assert(len(mode_range_min)==len(param_list))
    assert(len(mode_range_max)==len(param_list))

    cell_spacing = [int(n) for n in args.cell_spacing.split(',')]
    print("cell_spacing - ", cell_spacing)
    assert(len(cell_spacing)==len(param_list))
    ngrid_pts = [int(n) for n in args.ngrid_pts.split(',')]
    print("ngrid_pts - ", ngrid_pts)
    assert(len(ngrid_pts)==len(param_list))

    #TODO: How to deal with multipliers to mode_ranges dependent on a function of other modes (e.g., sqrt(thread_count))
    cell_nodes = []
    num_grid_pts=1
    contract_str = ''
    for i in range(len(param_list)):
	cell_nodes.append(np.rint(generate_nodes(mode_range_min[i],mode_range_max[i],ngrid_pts[i],cell_spacing[i])))
        contract_str += 'r,'
	for j in range(1,len(cell_nodes[-1])):
	    if (cell_nodes[-1][j] <= cell_nodes[-1][j-1]):
		cell_nodes[-1][j] = cell_nodes[-1][j-1] + 1
	num_grid_pts *= len(cell_nodes[-1])
    contract_str = contract_str[:-1]
    print("cell_nodes - ", cell_nodes)
    print("num_grid_pts - ", num_grid_pts)
    print("contract_str - ", contract_str)

    tensor_map = range(len(param_list))
    if (args.tensor_map != ''):
	tensor_map = [int(n) for n in args.tensor_map.split(',')]
    tensor_mode_lengths = [1]*len(tensor_map)
    for i in range(len(param_list)):
	tensor_mode_lengths[tensor_map[i]] = len(cell_nodes[i])
    print("tensor mode lengths - ", tensor_mode_lengths)

    interp_map = [1]*len(param_list)
    if (args.interp_map != ''):
	interp_map = [int(n) for n in args.interp_map.split(',')]
    interp_modes = []
    for i in range(len(interp_map)):
        if interp_map[i]==1:
            interp_modes.append(i)

    omega_ctf = ctf.tensor(tuple(tensor_mode_lengths), sp=False)
    Tdense_ctf = ctf.tensor(tuple(tensor_mode_lengths), sp=False)

    timers = [0.]*4	# Tensor generation, ALS, Total CV, Total Test Set evaluation

    start_time = time.time()
    nodes = []
    density = 0.
    training_node_list = []
    training_data_list = []
    save_training_nodes = []
    node_data_dict = {}
    node_count_dict = {}
    #tfile = open("gemm-%dx%dx%d.csv"%(ngrid_pts[0],ngrid_pts[1],ngrid_pts[2]),'w')
    for i in range(training_set_size):
	input_tuple = training_inputs[i,:]
	node_key = []
	for j in range(len(input_tuple)):
	    node_key.append(get_node_index(input_tuple[j],cell_nodes[j],cell_spacing[j]))
	    if (node_key[-1] < 0):
		assert(0)
	save_training_nodes.append(node_key)
	if (tuple(node_key) not in node_data_dict):
	    node_count_dict[tuple(node_key)] = 1
	    node_data_dict[tuple(node_key)] = training_data[i]
	else:
	    node_count_dict[tuple(node_key)] += 1
	    node_data_dict[tuple(node_key)] += training_data[i]
    density = len(node_data_dict.keys())*1./num_grid_pts
    for key in node_count_dict.keys():
	training_node_list.append(key)
	training_data_list.append(node_data_dict[key]/node_count_dict[key])
        #tfile.write("%d,%d,%d,%d,%g\n"%(0,cell_nodes[0][key[0]],cell_nodes[1][key[1]],cell_nodes[2][key[2]],training_data_list[-1]))
    #tfile.close()
    node_data = np.array(training_data_list)

    omega_ctf.write(training_node_list,np.ones(len(training_node_list)))
    Tdense_ctf.write(training_node_list,node_data.reshape(len(training_node_list)))
    omega = omega_ctf.to_nparray()
    Tdense = Tdense_ctf.to_nparray()

    save_test_nodes = []
    for i in range(test_set_size):
	input_tuple = test_inputs[i,:]
        node_key = []
        for j in range(len(input_tuple)):
            node_key.append(get_node_index(input_tuple[j],cell_nodes[j],cell_spacing[j]))
        save_test_nodes.append(node_key)

    validation_nodes = save_test_nodes[:split_idx]
    test_nodes = save_test_nodes[split_idx:]
    test_set_size = len(test_nodes)
    validation_inputs = test_inputs[:split_idx]
    validation_data = test_data[:split_idx]
    test_inputs = test_inputs[split_idx:split_idx+test_set_size,:]
    test_data = test_data[split_idx:split_idx+test_set_size]

    start_time = time.time()

    model_parameters = model_list[0]
    model_predictions = []
    true_results = []	# Needed because we may skip some entries due to unobserved grid-points
    for k in range(len(test_nodes)):
	input_tuple = test_inputs[k,:]*1.	# Note: without this cast from into to float, interpolation produces zeros
	node = test_nodes[k]
	midpoints = []
	for j in range(len(interp_modes)):
            cell_node_idx = interp_modes[j]
	    midpoints.append(get_midpoint(get_cell_index(input_tuple[interp_modes[j]],cell_nodes[cell_node_idx]), cell_nodes[cell_node_idx], cell_spacing[interp_modes[j]]))
	element_index_modes_list = []
	for j in range(len(interp_modes)):
            cell_node_idx = interp_modes[j]
	    element_index_modes_list.append([])
	    for xx in range(model_parameters[3]):
		if (input_tuple[interp_modes[j]] <= midpoints[j]):
		    element_index_modes_list[-1].append(node[interp_modes[j]]-(model_parameters[3]-1)/2+xx)
		else:
		    element_index_modes_list[-1].append(node[interp_modes[j]]-model_parameters[3]/2+xx)
	    if (element_index_modes_list[-1][0]<0):
		offset = element_index_modes_list[-1][0]*(-1)
		for xx in range(model_parameters[3]):
		    element_index_modes_list[-1][xx] += offset
	    if (element_index_modes_list[-1][-1]>=len(cell_nodes[cell_node_idx])):
		offset = element_index_modes_list[-1][-1]+1-len(cell_nodes[cell_node_idx])
		for xx in range(model_parameters[3]):
		    element_index_modes_list[-1][xx] -= offset
	model_val = 0.
        is_in_bounding_box = True	# Do not assess interpolation error if not all grid-points in bounding box around test_nodes[k] is observed.
	for j in range(model_parameters[3]**len(interp_modes)):
	    interp_id = j
	    interp_id_list = [0]*len(interp_modes)
	    counter = 0
	    while (interp_id>0):
		interp_id_list[counter] = interp_id%model_parameters[3]
		interp_id /= model_parameters[3]
		counter += 1
	    coeff = 1
	    for l in range(len(interp_modes)):
                cell_node_idx = interp_modes[l]
		for ll in range(model_parameters[3]):
		    if (ll != interp_id_list[l]):
			coeff *= (input_tuple[interp_modes[l]]-cell_nodes[cell_node_idx][element_index_modes_list[l][ll]])\
				 /(cell_nodes[cell_node_idx][element_index_modes_list[l][interp_id_list[l]]]-cell_nodes[cell_node_idx][element_index_modes_list[l][ll]])
            interp_counter = 0
            elem_indices = []
	    for l in range(len(input_tuple)):
		if (interp_map[l]==1):
		    #elem_indices.append(cell_nodes[l][element_index_modes_list[interp_counter][interp_id_list[interp_counter]]])
		    elem_indices.append(element_index_modes_list[interp_counter][interp_id_list[interp_counter]])
                    interp_counter += 1
		else:
		    elem_indices.append(node[l])
            if (omega[elem_indices[0],elem_indices[1],elem_indices[2]]==0):
                is_in_bounding_box = False
                break
	    if (numpy_eval == 1):
		t_val = Tdense[elem_indices[0],elem_indices[1],elem_indices[2]]
	    else:
		t_val = Tdense[elem_indices[0],elem_indices[1],elem_indices[2]]
	    model_val += coeff * t_val

        if (is_in_bounding_box == True):
	    model_predictions.append(model_val)
            true_results.append(test_data[k])

    timers[3] += (time.time()-start_time)

    test_error_metrics = [0]*12
    prediction_errors = [[] for k in range(3)]	# metrics 0 and 1 borrow, metrics 2 and 3 borrow, and rsme has already been calculated, so can reduce from 6 to 3
    for k in range(len(model_predictions)):
	prediction_errors[0].append(np.log((model_predictions[k] if model_predictions[k] > 0 else 1e-14)/true_results[k]))
	prediction_errors[1].append(np.abs(model_predictions[k]-true_results[k])/true_results[k])
	if (prediction_errors[1][-1] <= 0):
	    prediction_errors[1][-1] = 1e-14
	prediction_errors[2].append(np.abs(model_predictions[k]-true_results[k])/np.average([model_predictions[k],true_results[k]]))
    test_error_metrics[0] = np.average(prediction_errors[0])
    test_error_metrics[1] = np.std(prediction_errors[0],ddof=1)
    test_error_metrics[2] = np.average(np.absolute(prediction_errors[0]))
    test_error_metrics[3] = np.std(np.absolute(prediction_errors[0]),ddof=1)
    test_error_metrics[4] = np.average(np.asarray(prediction_errors[0])**2)
    test_error_metrics[5] = np.std(np.asarray(prediction_errors[0])**2,ddof=1)
    test_error_metrics[6] = scst.gmean(prediction_errors[1])
    test_error_metrics[7] = np.exp(np.std(np.log(prediction_errors[1]),ddof=1))
    test_error_metrics[8] = np.average(prediction_errors[1])
    test_error_metrics[9] = np.std(prediction_errors[1],ddof=1)
    test_error_metrics[10] = np.average(prediction_errors[2])
    test_error_metrics[11] = np.std(prediction_errors[2],ddof=1)

    """
    for k in range(len(prediction_errors[0])):
        print(np.absolute(prediction_errors[0][k]))
    """

    columns = (\
        "input:training_set_size",\
        "input:test_set_size",\
        "input:tensor_dim",\
        "input:ngrid_pts",\
        "input:cell_spacing",\
        "input:density",\
        "input:response_transform",\
        "input:reg",\
        "input:nals_sweeps",\
        "input:cp_rank",\
        "input:interp_map",\
	"error:mlogq",\
	"error:mlogq2",\
	"error:gmre",\
	"error:mape",\
	"error:smape",\
        "time:tensor_generation",\
        "time:model_configuration",\
    )
    test_results_dict = {0:{\
        columns[0] : training_set_size,\
        columns[1] : len(model_predictions),\
        columns[2] : len(tensor_mode_lengths),\
        columns[3] : "-".join([str(n) for n in ngrid_pts]),\
        columns[4] : "-".join([str(n) for n in cell_spacing]),\
        columns[5] : density,\
        columns[6] : args.response_transform,\
        columns[7] : model_parameters[0],\
        columns[8] : model_parameters[1],\
        columns[9] : model_parameters[2],\
        columns[10] : "-".join([str(n) for n in interp_map]),\
	columns[11] : test_error_metrics[2],\
	columns[12] : test_error_metrics[4],\
	columns[13] : test_error_metrics[6],\
	columns[14] : test_error_metrics[8],\
	columns[15] : test_error_metrics[10],\
        columns[16] : timers[0],\
        columns[17] : timers[2],\
    } }
    test_results_df = pd.DataFrame(data=test_results_dict,index=columns).T
    test_results_df.to_csv("%s"%(args.output_file),sep=',',header=1,mode="a")
