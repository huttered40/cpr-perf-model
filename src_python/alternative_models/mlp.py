import os, sys

def main(args):

    import time
    import numpy as np
    import pandas as pd

    from sklearn.neural_network import MLPRegressor

    sys.path.insert(0,'%s/../'%(os.getcwd()))
    from util import extract_datasets, get_error_metrics,write_statistics_to_file,get_model_size, transform_dataset, transform_predictor, transform_response, inverse_transform_response

    training_df = pd.read_csv(args.training_file, index_col=0, sep=',')
    test_df = pd.read_csv(args.test_file, index_col=0, sep=',')
    param_list = training_df.columns[args.input_columns].tolist()
    data_list = training_df.columns[args.data_columns].tolist()

    (training_configurations,training_data,training_set_size,test_configurations,test_data,test_set_size,mode_range_min,mode_range_max)\
      = extract_datasets(training_df,test_df,param_list,data_list,args.training_set_size,args.test_set_size,args.mode_range_min,args.mode_range_max)
    training_configurations,training_data = transform_dataset(args.predictor_transform_type,args.response_transform_type,training_configurations,training_data)

    timers = []
    start_time = time.time()
    model = MLPRegressor(hidden_layer_sizes=args.hidden_layer_sizes,activation=args.activation_function, solver=args.solver, random_state=0)
    model.fit(training_configurations,training_data)
    timers.append(time.time()-start_time)

    model_size = get_model_size(model,"Model.joblib")

    start_time = time.time()
    model_predictions = []
    for k in range(test_set_size):
        configuration = transform_predictor(args.predictor_transform_type,test_configurations[k,:]*1.)
        model_predictions.append(inverse_transform_response(args.response_transform_type,nn_model.predict([configuration])[0]))
    test_error_metrics = get_error_metrics(test_set_size,test_configurations,test_data,model_predictions,args.print_test_error)
    timers.append(time.time()-start_time)

    start_time = time.time()
    model_predictions = []
    for k in range(training_set_size):
        configuration = training_configurations[k,:]*1.
        model_predictions.append(inverse_transform_response(args.response_transform_type,nn_model.predict([configuration])[0]))
    training_error_metrics = get_error_metrics(training_set_size,training_configurations,inverse_transform_response(args.response_transform_type,training_data),model_predictions,0)
    timers.append(time.time()-start_time)
    write_statistics_to_file(args.output_file,test_error_metrics,training_error_metrics,timers,[training_set_size,test_set_size],model_size,[args.hidden_layer_sizes],["model:hidden_layer_sizes"])    

if __name__ == "__main__":

    import argparse

    sys.path.insert(0,'%s/../'%(os.getcwd()))
    from arg_defs import add_shared_parameters

    parser = argparse.ArgumentParser()
    add_shared_parameters(parser)
    parser.add_argument(
        '--hidden-layer-sizes',
        type=int,
        nargs='+',
        default=[64,64],
        help='Number of units in each layer of a MLP.')
    parser.add_argument(
        '--activation-function',
        type=str,
        default='relu',
        help='Activation function for a MLP.')
    parser.add_argument(
        '--solver',
        type=str,
        default='adam',
        help='Solver for optimizing a MLP.')
    args, _ = parser.parse_known_args()
    main(args)
