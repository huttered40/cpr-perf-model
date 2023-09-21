import os, time, sys, copy
import numpy as np
import numpy.linalg as la
import pandas as pd
import argparse
import arg_defs as arg_defs

from sklearn import datasets, ensemble

sys.path.insert(0,'%s/../'%(os.getcwd()))
from util import extract_datasets, get_error_metrics,write_statistics_to_file,get_model_size,\
                 transform_dataset, transform_predictor, transform_response, inverse_transform_response

def generate_models(_ntrees,_depth):
    model_list = []
    for i in _ntrees:
        for j in _depth:
            model_list.append([i,j])
    return model_list

def main(args):
    timers = [0.]*3
    training_df = pd.read_csv('%s'%(args.training_file), index_col=0, sep=',')
    test_df = pd.read_csv('%s'%(args.test_file), index_col=0, sep=',')
    param_list = training_df.columns[[int(n) for n in args.input_columns.split(',')]].tolist()
    data_list = training_df.columns[[int(n) for n in args.data_columns.split(',')]].tolist()

    predictor_transform=[int(n) for n in args.predictor_transform.split(',')]
    tree_depth=[int(n) for n in args.tree_depth.split(',')]
    ntrees=[int(n) for n in args.ntrees.split(',')]
    model_list = generate_models(ntrees,tree_depth)

    if (args.verbose == 1):
	print("Location of training data: %s"%(args.training_file))
	print("Location of test data: %s"%(args.test_file))
	print("Location of output data: %s"%(args.output_file))
	print("args.input_columns - ", args.input_columns)
	print("args.data_columns - ", args.data_columns)
	print("param_list: ", param_list)
	print(model_list)

    (training_configurations,training_data,training_set_size,\
            validation_configurations,validation_data,validation_set_size,\
            test_configurations,test_data,test_set_size,mode_range_min,mode_range_max)\
      = extract_datasets(training_df,test_df,param_list,data_list,args.training_set_size,\
          args.test_set_size,args.training_set_split_percentage,args.mode_range_min,args.mode_range_max)
    training_configurations,training_data = transform_dataset(predictor_transform,args.response_transform,training_configurations,training_data)

    start_time = time.time()
    opt_model_parameters = [-1]
    current_best_model = []
    opt_error_metrics = [100000.]*16
    for model_parameters in model_list:
	params = {
	"n_estimators": model_parameters[0],
	"max_depth": model_parameters[1],
	"min_samples_split": 5,
	"learning_rate": 0.01,
	#"loss": "squared_error",
	}
	gb_model = ensemble.GradientBoostingRegressor(**params)
        model_predictions = []
        start_time_solve = time.time()
	gb_model.fit(training_configurations,training_data)
        timers[0] += (time.time()-start_time_solve)
        # Now validate on validation set
	for k in range(validation_set_size):
	    configuration = transform_predictor(predictor_transform,validation_configurations[k,:]*1.)
            model_predictions.append(inverse_transform_response(args.response_transform,gb_model.predict([configuration])[0]))
        validation_error_metrics = get_error_metrics(validation_set_size,validation_configurations,validation_data,model_predictions,args.print_test_error)
	if (validation_error_metrics[2] < opt_error_metrics[1]):
	    opt_model_parameters = copy.deepcopy(model_parameters)
	    opt_error_metrics = copy.deepcopy(validation_error_metrics)
	    current_best_model = copy.deepcopy(gb_model)
    if (validation_set_size > 0):
        GB_Model = current_best_model
    else:
        GB_Model = gb_model
        opt_model_parameters = model_parameters
    timers[1] += (time.time()-start_time)

    model_size = get_model_size(GB_Model,"GB_Model.joblib")

    start_time = time.time()
    model_predictions = []
    for k in range(test_set_size):
        configuration = transform_predictor(predictor_transform,test_configurations[k,:]*1.)
        model_predictions.append(inverse_transform_response(args.response_transform,GB_Model.predict([configuration])[0]))
    timers[2] += (time.time()-start_time)
    test_error_metrics = get_error_metrics(test_set_size,test_configurations,test_data,model_predictions,args.print_test_error)

    model_predictions = []
    for k in range(training_set_size):
        configuration = training_configurations[k,:]*1.
        model_predictions.append(inverse_transform_response(args.response_transform,GB_Model.predict([configuration])[0]))
    training_error_metrics = get_error_metrics(training_set_size,training_configurations,inverse_transform_response(args.response_transform,training_data),model_predictions,0)
    write_statistics_to_file(args.output_file,test_error_metrics,training_error_metrics,timers,[training_set_size,validation_set_size,test_set_size],model_size,[opt_model_parameters[0],opt_model_parameters[1]],["model:ntrees","model:max_tree_depth"])    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    arg_defs.add_general_arguments(parser)
    args, _ = parser.parse_known_args()
    main(args)
