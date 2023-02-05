import ctf
import time,os,sys,joblib,copy,argparse
import numpy as np
import numpy.linalg as la
import scipy.stats as scst
import random as rand
import pandas as pd
import arg_defs as arg_defs

from cpr_model import cpr_model
from util import extract_datasets, get_error_metrics,write_statistics_to_file,get_model_size,\
                 transform_dataset, transform_predictor, transform_response, inverse_transform_response

glob_comm = ctf.comm()

def generate_models(reg,cp_rank):
    model_list = []
    for i in reg:
        for j in cp_rank:
            model_list.append((i,j))
    return model_list

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    arg_defs.add_general_arguments(parser)
    args, _ = parser.parse_known_args()

    timers = [0.]*3
    training_df = pd.read_csv('%s'%(args.training_file), index_col=0, sep=',')
    test_df = pd.read_csv('%s'%(args.test_file), index_col=0, sep=',')
    param_list = training_df.columns[[int(n) for n in args.input_columns.split(',')]].tolist()
    data_list = training_df.columns[[int(n) for n in args.data_columns.split(',')]].tolist()

    regularization_parameters = [float(n) for n in args.reg.split(',')]
    cp_ranks = [int(n) for n in args.cp_rank.split(',')]

    # Generate list of model types parameterized on hyper-parameters
    model_list = generate_models(regularization_parameters,cp_ranks)

    (training_configurations,training_data,training_set_size,\
            validation_configurations,validation_data,validation_set_size,\
            test_configurations,test_data,test_set_size,mode_range_min,mode_range_max)\
      = extract_datasets(training_df,test_df,param_list,data_list,args.training_set_size,\
          args.test_set_size,args.training_set_split_percentage,args.mode_range_min,args.mode_range_max)

    cell_spacing = [int(n) for n in args.cell_spacing.split(',')]
    assert(len(cell_spacing)==len(param_list))
    ngrid_pts = [int(n) for n in args.ngrid_pts.split(',')]
    assert(len(ngrid_pts)==len(param_list))

    if (args.print_diagnostics == 1):
	print("Location of training data: %s"%(args.training_file))
	print("Location of test data: %s"%(args.test_file))
	print("Location of output data: %s"%(args.output_file))
	print("args.input_columns: ",args.input_columns)
	print("args.data_columns: ",args.data_columns)
	print("param_list: ", param_list)
	print("reg: ",regularization_parameters)
	print("cp_rank: ",cp_ranks)
	print("model_list: ", model_list)
	print("cell_spacing: ", cell_spacing)
	print("ngrid_pts: ", ngrid_pts)

    start_time = time.time()
    opt_model_parameters = [-1,-1]
    opt_model = []
    opt_model_info = [np.inf]*4
    opt_error_metrics = [100000.]*16
    # Iterate over all model hyper-parameters (not including cell-count)
    for model_parameters in model_list:
        cp_rank = model_parameters[1]
        reg_lambda = model_parameters[0]
        start_time_solve = time.time()
        cpr_mod = cpr_model(cp_rank,args.cp_rank_for_extrapolation,args.loss_function,reg_lambda,args.max_spline_degree,args.interp_map,\
                    args.response_transform,args.sweep_tol,args.max_num_sweeps,\
                    args.tol_newton,args.max_num_newton_iter,args.barrier_start,args.barrier_stop,\
                    args.barrier_reduction_factor,cell_spacing,ngrid_pts,mode_range_min,mode_range_max,
                    args.build_extrapolation_model)
        num_tensor_elements,density,loss1,loss2 = cpr_mod.fit(training_configurations,training_data)
        timers[0] += (time.time()-start_time_solve)
        model_predictions = []
	# Evaluate on validation set
	for k in range(validation_set_size):
	    configuration = validation_configurations[k,:]*1.
            model_predictions.append(cpr_mod.predict(configuration))
        validation_error_metrics = get_error_metrics(validation_set_size,validation_configurations,validation_data,model_predictions)
	model_info = [num_tensor_elements,density,loss1,loss2]
	if (validation_set_size==0 or validation_error_metrics[2] < opt_error_metrics[2]):
            opt_model_parameters = copy.deepcopy(model_parameters)
	    opt_model_info = copy.deepcopy(model_info)
	    opt_model = copy.deepcopy(cpr_mod)
    timers[1] += (time.time()-start_time)

    model_size = get_model_size(opt_model,"CPR_Model.joblib")

    if (args.print_model_parameters):
        opt_model.print_parameters()

    start_time = time.time()
    model_predictions = []
    for k in range(test_set_size):
	configuration = test_configurations[k,:]*1.
        model_predictions.append(opt_model.predict(configuration))
    timers[2] += (time.time()-start_time)
    test_error_metrics = get_error_metrics(test_set_size,test_configurations,test_data,model_predictions,args.print_test_error)
    model_predictions = []
    for k in range(training_set_size):
	configuration = training_configurations[k,:]*1.
	model_predictions.append(opt_model.predict(configuration))
    training_error_metrics = get_error_metrics(training_set_size,training_configurations,training_data,model_predictions,0)
    write_statistics_to_file(args.output_file,test_error_metrics,training_error_metrics,timers,[training_set_size,validation_set_size,test_set_size],model_size,[len(ngrid_pts),opt_model_parameters[0],opt_model_parameters[1],opt_model_info[0],opt_model_info[1],opt_model_info[2],opt_model_info[3],args.cp_rank_for_extrapolation,args.loss_function,args.sweep_tol,args.max_num_sweeps,args.tol_newton,args.max_num_newton_iter,args.barrier_start,args.barrier_stop,args.barrier_reduction_factor],["model:tensor_dim","model:reg","model:cprank","model:ntensor_elems","model:tensor_density","model:loss1","model:loss2","extrap_cp_rank","loss_function","sweep_tol","max_num_sweeps","tol_newton","max_num_newton_iter","barrier_start","barrier_stop","barrier_reduction_factor"])    
