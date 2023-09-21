import os, time, sys, copy
import numpy as np
import pandas as pd
import argparse
import arg_defs as arg_defs

from sklearn.neural_network import MLPRegressor

sys.path.insert(0,'%s/../'%(os.getcwd()))
from util import extract_datasets, get_error_metrics,write_statistics_to_file,get_model_size,\
                 transform_dataset, transform_predictor, transform_response, inverse_transform_response

def main(args):
    timers = [0.]*3
    training_df = pd.read_csv('%s'%(args.training_file), index_col=0, sep=',')
    test_df = pd.read_csv('%s'%(args.test_file), index_col=0, sep=',')
    param_list = training_df.columns[[int(n) for n in args.input_columns.split(',')]].tolist()
    data_list = training_df.columns[[int(n) for n in args.data_columns.split(',')]].tolist()

    predictor_transform=[int(n) for n in args.predictor_transform.split(',')]
    hidden_layer_sizes = [int(n) for n in args.hidden_layer_sizes.split(',')]
    model_list = [hidden_layer_sizes]

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
	nn_model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,activation=args.activation, solver=args.solver, random_state=0)
        model_predictions = []
        start_time_solve = time.time()
	nn_model.fit(training_configurations,training_data)
        timers[0] += (time.time()-start_time_solve)
        # Now validate on validation set
	for k in range(validation_set_size):
	    configuration = transform_predictor(predictor_transform,validation_configurations[k,:]*1.)
            model_predictions.append(inverse_transform_response(args.response_transform,nn_model.predict([configuration])[0]))
        validation_error_metrics = get_error_metrics(validation_set_size,validation_configurations,validation_data,model_predictions)
	if (validation_error_metrics[2] < opt_error_metrics[1]):
	    opt_model_parameters = copy.deepcopy(model_parameters)
	    opt_error_metrics = copy.deepcopy(validation_error_metrics)
	    current_best_model = copy.deepcopy(nn_model)
    if (validation_set_size > 0):
        NN_Model = current_best_model
    else:
        NN_Model = nn_model
        opt_model_parameters = model_parameters
    timers[1] += (time.time()-start_time)

    model_size = get_model_size(NN_Model,"NN_Model.joblib")

    start_time = time.time()
    model_predictions = []
    for k in range(test_set_size):
        configuration = transform_predictor(predictor_transform,test_configurations[k,:]*1.)
        model_predictions.append(inverse_transform_response(args.response_transform,NN_Model.predict([configuration])[0]))
    timers[2] += (time.time()-start_time)
    test_error_metrics = get_error_metrics(test_set_size,test_configurations,test_data,model_predictions,args.print_test_error)

    model_predictions = []
    for k in range(training_set_size):
        configuration = training_configurations[k,:]*1.
        model_predictions.append(inverse_transform_response(args.response_transform,NN_Model.predict([configuration])[0]))
    training_error_metrics = get_error_metrics(training_set_size,training_configurations,inverse_transform_response(args.response_transform,training_data),model_predictions,0)
    write_statistics_to_file(args.output_file,test_error_metrics,training_error_metrics,timers,[training_set_size,validation_set_size,test_set_size],model_size,[args.hidden_layer_sizes],["model:hidden_layer_sizes"])    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    arg_defs.add_general_arguments(parser)
    parser.add_argument(
        '--hidden-layer-sizes',
        type=str,
        default='64,64',
        metavar='str',
        help='Comma-delimited list signifying the number of units in each layer of a MLP (default: 64,64).')
    parser.add_argument(
        '--activation',
        type=str,
        default='relu',
        metavar='str',
        help='Activation function for a MLP (default: relu).')
    parser.add_argument(
        '--solver',
        type=str,
        default='adam',
        metavar='str',
        help='Solver for optimizing a MLP (default: adam)')
    args, _ = parser.parse_known_args()
    main(args)
