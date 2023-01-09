import ctf
import time,os,sys,joblib,copy,argparse
import numpy as np
import numpy.linalg as la
import scipy.stats as scst
import random as rand
import pandas as pd
import arg_defs as arg_defs

from cpr_model import cpr_model
from util import extract_datasets,get_error_metrics

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

    training_df = pd.read_csv('%s'%(args.training_file), index_col=0, sep=',')
    test_df = pd.read_csv('%s'%(args.test_file), index_col=0, sep=',')
    param_list = training_df.columns[[int(n) for n in args.input_columns.split(',')]].tolist()
    data_list = training_df.columns[[int(n) for n in args.data_columns.split(',')]].tolist()

    regularization_parameters = [float(n) for n in args.reg.split(',')]
    cp_ranks = [int(n) for n in args.cp_rank.split(',')]

    # Generate list of model types parameterized on hyper-parameters
    model_list = generate_models(regularization_parameters,cp_ranks)

    (training_inputs,training_data,training_set_size,\
            validation_inputs,validation_data,validation_set_size,\
            test_inputs,test_data,test_set_size,mode_range_min,mode_range_max)\
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

    timers = [0.]*4	# Tensor generation, ALS, Total CV, Total Test Set evaluation
    opt_model_parameters = [-1,-1]
    opt_model = []
    opt_model_info = [np.inf]*4
    # Iterate over all model hyper-parameters (not including cell-count)
    for model_parameters in model_list:
        cp_rank = model_parameters[1]
        reg_lambda = model_parameters[0]
        start_time = time.time()
        cpr_mod = cpr_model(cp_rank,reg_lambda,args.max_spline_degree,args.interp_map,\
                    args.response_transform,args.sweep_tol,args.max_num_sweeps,\
                    args.tol_newton,args.max_num_newton_iter,args.barrier_start,\
                    args.barrier_reduction_factor,cell_spacing,ngrid_pts,mode_range_min,mode_range_max,
                    args.build_extrapolation_model)
        num_tensor_elements,density,loss1,loss2 = cpr_mod.fit(training_inputs,training_data)
        timers[0] += (time.time()-start_time)

        model_predictions = []
	# Evaluate on validation set
	for k in range(validation_set_size):
	    input_tuple = validation_inputs[k,:]*1.
            model_predictions.append(cpr_mod.predict(input_tuple))
        validation_error_metrics = get_error_metrics(validation_set_size,validation_inputs,validation_data,model_predictions)
	model_info = [num_tensor_elements,density,loss1,loss2]
	#print("Validation Error for (reg=%f,rank=%d) is "%(model_parameters[0],model_parameters[1]),validation_error_metrics, " with time: ", timers)
	if (validation_set_size==0 or validation_error_metrics[2] < opt_error_metrics[2]):
            opt_model_parameters = copy.deepcopy(model_parameters)
	    opt_model_info = copy.deepcopy(model_info)
	    opt_model = copy.deepcopy(cpr_mod)
    timers[2] += (time.time()-start_time)

    joblib.dump(opt_model, "CPR_Model.joblib") 
    model_size = os.path.getsize('CPR_Model.joblib')
    print("CPR model size: %f bytes"%(model_size))

    if (args.print_model_parameters):
        opt_model.print_parameters()

    start_time = time.time()
    model_predictions = []
    for k in range(test_set_size):
	input_tuple = test_inputs[k,:]*1.
        model_val = opt_model.predict(input_tuple)
        model_predictions.append(model_val)
    timers[3] += (time.time()-start_time)
    test_error_metrics = get_error_metrics(test_set_size,test_inputs,test_data,model_predictions)
    if (args.check_training_error):
        for k in range(training_set_size):
	    input_tuple = training_inputs[k,:]*1.
            model_val = opt_model.predict(input_tuple)
            model_predictions.append(model_val)
        training_error_metrics = get_error_metrics(training_set_size,training_inputs,training_data,model_predictions)

    # Write relevant error statistics to file
    columns = (\
        "input:training_set_size",\
        "input:test_set_size",\
        "input:tensor_dim",\
        "input:response_transform",\
        "input:reg",\
        "input:cp_rank",\
        "model_size",\
        "num_tensor_elements",\
        "tensor_density",\
        "error:loss1",\
        "error:loss2",\
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
        columns[1] : test_set_size,\
        columns[2] : len(ngrid_pts),\
        columns[3] : args.response_transform,\
        columns[4] : opt_model_parameters[0],\
        columns[5] : opt_model_parameters[1],\
        columns[6] : model_size,\
	columns[7] : opt_model_info[0],\
	columns[8] : opt_model_info[1],\
	columns[9] : opt_model_info[2],\
	columns[10] : opt_model_info[3],\
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
