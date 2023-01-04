import time, sys, copy
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import numpy.linalg as la
import scipy.stats as scst
import random as rand
import os
import pandas as pd
import argparse
import arg_defs as arg_defs
import joblib

def generate_models(_ntrees,_depth):
    model_list = []
    for i in _ntrees:
        for j in _depth:
            model_list.append([i,j])
    return model_list

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    arg_defs.add_general_arguments(parser)
    args, _ = parser.parse_known_args()

    timers = [0.]*3	# Matrix generation, Solve, Total CV, Total Test Set evaluation
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

    tree_depth=[int(n) for n in args.tree_depth.split(',')]
    ntrees=[int(n) for n in args.ntrees.split(',')]
    # Note: no option to transform data because basis functions assume certain structure,
    #   and transforming the runtimes would necessitate transforming the basis functions.
    # Generate list of model types parameterized on hyper-parameters
    model_list = generate_models(ntrees,tree_depth)
    print(model_list)

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

    if (args.kernel_name == "kripke"):
        for i in range(len(training_data)):
	    input_tuple = training_inputs[i,:]
            for j in range(len(input_tuple)):
                if (param_list[j]=="layout"):
                    input_tuple[j] = layout_dict[input_tuple[j]]
            training_inputs[i,:] = input_tuple
        for i in range(len(test_data)):
	    input_tuple = test_inputs[i,:]
            for j in range(len(input_tuple)):
                if (param_list[j]=="layout"):
                    input_tuple[j] = layout_dict[input_tuple[j]]
            test_inputs[i,:] = input_tuple
    if (args.kernel_name == "exafmm"):
        for i in range(len(training_data)):
	    input_tuple = training_inputs[i,:]
            for j in range(len(input_tuple)):
                if (param_list[j]=="level"):
                    input_tuple[j] = input_tuple[j] + 1 # Add 1 so that log(..) can be used
                elif (param_list[j]=="order"):
                    input_tuple[j] = input_tuple[j] + 1 # Add 1 so that log(..) can be used
            training_inputs[i,:] = input_tuple
        for i in range(len(test_data)):
	    input_tuple = test_inputs[i,:]
            for j in range(len(input_tuple)):
                if (param_list[j]=="level"):
                    input_tuple[j] = input_tuple[j] + 1 # Add 1 so that log(..) can be used
                elif (param_list[j]=="order"):
                    input_tuple[j] = input_tuple[j] + 1 # Add 1 so that log(..) can be used
            test_inputs[i,:] = input_tuple

    start_time = time.time()
    opt_model_parameters = [-1]
    current_best_model = []
    opt_error_metrics = [100000.]*6	# arithmetic sum of log Q, arithmetic sum of log^2 Q, geometric mean of relative errors, MAPE, SMAPE, RMSE
    for model_parameters in model_list:
	rf_model = RandomForestRegressor(n_estimators=model_parameters[0],max_depth=model_parameters[1],random_state=0)
        model_predictions = []
        start_time_solve = time.time()
        if (args.predictor_transform==1 and args.response_transform==1):
	    rf_model.fit(np.log(training_inputs),np.log(training_data))
        elif (args.predictor_transform==0 and args.response_transform==1):
	    rf_model.fit(training_inputs,np.log(training_data))
        elif (args.predictor_transform==1 and args.response_transform==0):
	    rf_model.fit(np.log(training_inputs),training_data)
        elif (args.predictor_transform==0 and args.response_transform==0):
	    rf_model.fit(training_inputs,training_data)

        joblib.dump(rf_model, "RF_model.joblib") 
	model_size = os.path.getsize('RF_model.joblib')
        print("RF model size: %f bytes"%(model_size))

        timers[0] += (time.time()-start_time_solve)
        # Now validate on validation set
	for k in range(validation_set_size):
	    input_tuple = test_inputs[k,:]*1. if args.predictor_transform==0 else np.log(test_inputs[k,:]*1.) # Note: without this cast from into to float, interpolation produces zeros
            # Box-cox: input_tuple = np.power([test_inputs[3*i],test_inputs[3*i+1],test_inputs[3*i+2]],1./np.absolute(data_transformation))
            model_val = rf_model.predict([input_tuple])[0]
            if (args.response_transform == 1):
                model_val = np.exp(model_val)
            model_predictions.append(model_val)

        validation_error_metrics = [0]*10
        prediction_errors = [[] for k in range(3)]
	for k in range(validation_set_size):
            input_tuple = test_inputs[k,:]*1.
            prediction_errors[0].append(np.log((model_predictions[k] if model_predictions[k] > 0 else 1e-14)/test_data[k]))
            prediction_errors[1].append(np.abs(model_predictions[k]-test_data[k])/test_data[k])
	    if (prediction_errors[1][-1] <= 0):
		prediction_errors[1][-1] = 1e-14
            prediction_errors[2].append(np.abs(model_predictions[k]-test_data[k])/np.average([model_predictions[k],test_data[k]]))
	if (validation_set_size > 0):
	    validation_error_metrics[0] = np.average(prediction_errors[0])
	    validation_error_metrics[1] = np.std(prediction_errors[0],ddof=1)
	    validation_error_metrics[2] = np.average(np.asarray(prediction_errors[0])**2)
	    validation_error_metrics[3] = np.std(np.asarray(prediction_errors[0])**2,ddof=1)
	    validation_error_metrics[4] = scst.gmean(prediction_errors[1])
	    validation_error_metrics[5] = np.exp(np.std(np.log(prediction_errors[1]),ddof=1))
	    validation_error_metrics[6] = np.average(prediction_errors[1])
	    validation_error_metrics[7] = np.std(prediction_errors[1],ddof=1)
	    validation_error_metrics[8] = np.average(prediction_errors[2])
	    validation_error_metrics[9] = np.std(prediction_errors[2],ddof=1)
	    print("Validation Error for (degree=%d) is "%(model_parameters[0]),validation_error_metrics, " with time: ", timers)
	    if (validation_error_metrics[2] < opt_error_metrics[1]):
		opt_model_parameters = copy.deepcopy(model_parameters)
		opt_error_metrics = copy.deepcopy(validation_error_metrics)
		current_best_model = copy.deepcopy(rf_model)
    if (validation_set_size > 0):
        RF_Model = current_best_model
    else:
        RF_Model = rf_model
        opt_model_parameters = model_parameters
    timers[1] += (time.time()-start_time)

    start_time = time.time()
    model_predictions = []
    for k in range(test_set_size):
	input_tuple = test_inputs[split_idx+k,:]*1. if args.predictor_transform==0 else np.log(test_inputs[split_idx+k,:]*1.) # Note: without this cast from into to float, interpolation produces zeros
	# Box-cox: input_tuple = np.power([test_inputs[3*i],test_inputs[3*i+1],test_inputs[3*i+2]],1./np.absolute(data_transformation))
	model_val = RF_Model.predict([input_tuple])[0]
	if (args.response_transform == 1):
	    model_val = np.exp(model_val)
	model_predictions.append(model_val)
    timers[2] += (time.time()-start_time)

    test_error_metrics = [0]*12
    prediction_errors = [[] for k in range(3)]
    for k in range(test_set_size):
        input_tuple = test_inputs[split_idx+k,:]*1.
        #if (args.kernel_name == "snap" and np.log((model_predictions[k] if model_predictions[k] > 0 else 1e-14)/test_data[split_idx+k])**2 > 2):
        #    continue
	prediction_errors[0].append(np.log((model_predictions[k] if model_predictions[k] > 0 else 1e-14)/test_data[split_idx+k]))
	prediction_errors[1].append(np.abs(model_predictions[k]-test_data[split_idx+k])/test_data[split_idx+k])
	#if (input_tuple[0]>512 and input_tuple[1]>512 and input_tuple[2]>512):
	#    print("test error - ", input_tuple, test_data[k], model_predictions[k], prediction_errors[0][-1]**2) 
        if (prediction_errors[1][-1] <= 0):
            prediction_errors[1][-1] = 1e-14
	prediction_errors[2].append(np.abs(model_predictions[k]-test_data[split_idx+k])/np.average([model_predictions[k],test_data[split_idx+k]]))
    test_error_metrics[0] = np.average(prediction_errors[0])
    test_error_metrics[1] = np.std(prediction_errors[0],ddof=1)
    test_error_metrics[2] = np.average(np.absolute(np.asarray(prediction_errors[0])))
    test_error_metrics[3] = np.std(np.absolute(np.asarray(prediction_errors[0])),ddof=1)
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
        "input:predictor_transform",\
        "input:response_transform",\
        "ntrees",\
        "max_tree_depth",\
        "model_size",\
 	"error:mlogq",\
	"error:mlogq2",\
	"error:gmre",\
	"error:mape",\
	"error:smape",\
        "time:model_configuration",\
        "time:model_configuration+validation",\
        "time:prediction",\
    )
    test_results_dict = {0:{\
        columns[0] : training_set_size,\
        columns[1] : test_set_size,\
        columns[2] : args.predictor_transform,\
        columns[3] : args.response_transform,\
        columns[4] : opt_model_parameters[0],\
        columns[5] : opt_model_parameters[1],\
        columns[6] : model_size,\
	columns[7] : test_error_metrics[2],\
	columns[8] : test_error_metrics[4],\
	columns[9] : test_error_metrics[6],\
	columns[10] : test_error_metrics[8],\
	columns[11] : test_error_metrics[10],\
        columns[12] : timers[0],\
        columns[13] : timers[1],\
        columns[14] : timers[2],\
    } }
    test_results_df = pd.DataFrame(data=test_results_dict,index=columns).T
    test_results_df.to_csv("%s"%(args.output_file),sep=',',header=1,mode="a")
