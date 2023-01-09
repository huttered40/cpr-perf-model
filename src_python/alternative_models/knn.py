import os, joblib, time, sys, copy
import numpy as np
import numpy.linalg as la
import pandas as pd
import argparse
import arg_defs as arg_defs

from sklearn.neighbors import KNeighborsRegressor

sys.path.insert(0,'%s/../'%(os.getcwd()))
from util import extract_datasets, get_error_metrics

def generate_models(_nneighbors):
    model_list = []
    for i in _nneighbors:
        model_list.append([i])
    return model_list

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    arg_defs.add_general_arguments(parser)
    args, _ = parser.parse_known_args()

    timers = [0.]*3	# Matrix generation, Solve, Total CV, Total Test Set evaluation
    training_df = pd.read_csv('%s'%(args.training_file), index_col=0, sep=',')
    test_df = pd.read_csv('%s'%(args.test_file), index_col=0, sep=',')
    param_list = training_df.columns[[int(n) for n in args.input_columns.split(',')]].tolist()
    data_list = training_df.columns[[int(n) for n in args.data_columns.split(',')]].tolist()

    nneighbors=[int(n) for n in args.nneighbors.split(',')]
    # Note: no option to transform data because basis functions assume certain structure,
    #   and transforming the runtimes would necessitate transforming the basis functions.
    # Generate list of model types parameterized on hyper-parameters
    model_list = generate_models(nneighbors)

    (training_inputs,training_data,training_set_size,\
            validation_inputs,validation_data,validation_set_size,\
            test_inputs,test_data,test_set_size,mode_range_min,mode_range_max)\
      = extract_datasets(training_df,test_df,param_list,data_list,args.training_set_size,\
          args.test_set_size,args.training_set_split_percentage,args.mode_range_min,args.mode_range_max)

    if (args.print_diagnostics == 1):
	print("Location of training data: %s"%(args.training_file))
	print("Location of test data: %s"%(args.test_file))
	print("Location of output data: %s"%(args.output_file))
	print("args.input_columns - ", args.input_columns)
	print("args.data_columns - ", args.data_columns)
	print("param_list: ", param_list)
	print(model_list)

    start_time = time.time()
    opt_model_parameters = [-1]
    current_best_model = []
    opt_error_metrics = [100000.]*6	# arithmetic sum of log Q, arithmetic sum of log^2 Q, geometric mean of relative errors, MAPE, SMAPE, RMSE
    for model_parameters in model_list:
	knn_model = KNeighborsRegressor(n_neighbors=model_parameters[0],weights='distance')
        model_predictions = []
        start_time_solve = time.time()
        if (args.predictor_transform==1 and args.response_transform==1):
	    knn_model.fit(np.log(training_inputs),np.log(training_data))
        elif (args.predictor_transform==0 and args.response_transform==1):
	    knn_model.fit(training_inputs,np.log(training_data))
        elif (args.predictor_transform==1 and args.response_transform==0):
	    knn_model.fit(np.log(training_inputs),training_data)
        elif (args.predictor_transform==0 and args.response_transform==0):
	    knn_model.fit(training_inputs,training_data)

        joblib.dump(knn_model, "KNN_Model.joblib") 
        model_size = os.path.getsize('KNN_Model.joblib')
        print("KNN model size: %f bytes"%(model_size))

        timers[0] += (time.time()-start_time_solve)
        # Now validate on validation set
	for k in range(validation_set_size):
	    input_tuple = test_inputs[k,:]*1. if args.predictor_transform==0 else np.log(test_inputs[k,:]*1.) # Note: without this cast from into to float, interpolation produces zeros
            # Box-cox: input_tuple = np.power([test_inputs[3*i],test_inputs[3*i+1],test_inputs[3*i+2]],1./np.absolute(data_transformation))
            model_val = knn_model.predict([input_tuple])[0]
            if (args.response_transform == 1):
                model_val = np.exp(model_val)
            model_predictions.append(model_val)

        validation_error_metrics = get_error_metrics(validation_set_size,validation_inputs,validation_data,model_predictions)
	if (validation_error_metrics[2] < opt_error_metrics[1]):
	    opt_model_parameters = copy.deepcopy(model_parameters)
	    opt_error_metrics = copy.deepcopy(validation_error_metrics)
	    current_best_model = copy.deepcopy(knn_model)
    if (validation_set_size > 0):
        KNN_Model = current_best_model
    else:
        KNN_Model = knn_model
        opt_model_parameters = model_parameters
    timers[1] += (time.time()-start_time)

    start_time = time.time()
    model_predictions = []
    for k in range(test_set_size):
	input_tuple = test_inputs[k,:]*1. if args.predictor_transform==0 else np.log(test_inputs[k,:]*1.) # Note: without this cast from into to float, interpolation produces zeros
	# Box-cox: input_tuple = np.power([test_inputs[3*i],test_inputs[3*i+1],test_inputs[3*i+2]],1./np.absolute(data_transformation))
	model_val = KNN_Model.predict([input_tuple])[0]
	if (args.response_transform == 1):
	    model_val = np.exp(model_val)
	model_predictions.append(model_val)
    timers[2] += (time.time()-start_time)

    test_error_metrics = get_error_metrics(test_set_size,test_inputs,test_data,model_predictions)

    # Write relevant error statistics to file
    columns = (\
        "input:training_set_size",\
        "input:test_set_size",\
        "input:predictor_transform",\
        "input:response_transform",\
        "num_neighbors",\
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
        columns[5] : model_size,\
	columns[6] : test_error_metrics[2],\
	columns[7] : test_error_metrics[4],\
	columns[8] : test_error_metrics[6],\
	columns[9] : test_error_metrics[8],\
	columns[10] : test_error_metrics[10],\
        columns[11] : timers[0],\
        columns[12] : timers[1],\
        columns[13] : timers[2],\
    } }
    test_results_df = pd.DataFrame(data=test_results_dict,index=columns).T
    test_results_df.to_csv("%s"%(args.output_file),sep=',',header=1,mode="a")
