import os, sys

def main(args):

    import time
    import numpy as np
    import pandas as pd

    import pyearth as pe

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
    model = pe.Earth(max_degree=args.max_spline_degree,allow_linear=False)#,smooth=True))
    model.fit(training_configurations,training_data)
    timers.append(time.time()-start_time)
    #if (args.print_model_parameters == 1):
    print(model.summary())

    model_size = get_model_size(model,"Model.joblib")

    start_time = time.time()
    model_predictions = []
    for k in range(test_set_size):
        configuration = transform_predictor(args.predictor_transform_type,test_configurations[k,:]*1.)
        model_predictions.append(inverse_transform_response(args.response_transform_type,model.predict([configuration])[0]))
    test_error_metrics = get_error_metrics(test_set_size,test_configurations,test_data,model_predictions,args.print_test_error)
    timers.append(time.time()-start_time)

    start_time = time.time()
    model_predictions = []
    for k in range(training_set_size):
        configuration = training_configurations[k,:]*1.
        model_predictions.append(inverse_transform_response(args.response_transform_type,model.predict([configuration])[0]))
    training_error_metrics = get_error_metrics(training_set_size,training_configurations,inverse_transform_response(args.response_transform_type,training_data),model_predictions,0)
    timers.append(time.time()-start_time)
    write_statistics_to_file(args.output_file,test_error_metrics,training_error_metrics,timers,[training_set_size,test_set_size],model_size,[args.max_spline_degree],["model:max_spline_degree"])    

if __name__ == "__main__":

    import argparse

    sys.path.insert(0,'%s/../'%(os.getcwd()))
    from arg_defs import add_shared_parameters

    parser = argparse.ArgumentParser()
    add_shared_parameters(parser)
    parser.add_argument(
        '-J',
        '--max-spline-degree',
        type=int,
        default=3,
        help='Maximum spline degree.')
    args, _ = parser.parse_known_args()
    main(args)
