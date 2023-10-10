import os, sys
import numpy as np

def residuals(_model_coeffs, _responses, _features):
    _res = np.zeros((len(_responses),))
    for zz in range(len(_responses)):
        m = _features[zz,0]
        n = _features[zz,1]
        k = _features[zz,2]
        #_res[zz] = np.log(_responses[zz]) - np.log(_model_coeffs[0]*(i+_model_coeffs[1])*(j+_model_coeffs[2])*(k+_model_coeffs[3])+_model_coeffs[4] - _model_coeffs[0]*(i*1./_model_coeffs[1]+j*1./_model_coeffs[2]+k*1./_model_coeffs[3]+1)*_model_coeffs[1]*_model_coeffs[2]*_model_coeffs[3])
        _res[zz] = np.log(_responses[zz]) - np.log(_model_coeffs[0]*m*n*k + _model_coeffs[1]*(m*n + m*k + n*k + m*n*k/1000) + _model_coeffs[2])
    return _res

def generate_prediction(_model_coeffs,_input_tuple):
    i=_input_tuple[0]*1.
    j=_input_tuple[1]*1.
    k=_input_tuple[2]*1
    return _model_coeffs[0]*i*j*k + _model_coeffs[1]*(i*j + i*k + j*k) + _model_coeffs[2]

def main(args):

    import time
    import scipy.optimize as sco
    import pandas as pd

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
    model_coeffs = np.random.rand(3,)
    ret_val = sco.least_squares(residuals,model_coeffs,bounds=([0]*3,[np.inf]*3),args=(training_data,training_configurations))
    model_coeffs = ret_val.x
    timers.append(time.time()-start_time)

    model_size = model_list[0][0]

    start_time = time.time()
    model_predictions = []
    for k in range(test_set_size):
        configuration = transform_predictor(args.predictor_transform_type,test_configurations[k,:]*1.)
        model_predictions.append(inverse_transform_response(args.response_transform_type,generate_prediction(model_coeffs,configuration)))
    test_error_metrics = get_error_metrics(test_set_size,test_configurations,test_data,model_predictions,args.print_test_error)
    timers.append(time.time()-start_time)

    start_time = time.time()
    model_predictions = []
    for k in range(training_set_size):
        configuration = training_configurations[k,:]*1.
        model_predictions.append(inverse_transform_response(args.response_transform_type,generate_prediction(model_coeffs,configuration)))
    training_error_metrics = get_error_metrics(training_set_size,training_configurations,inverse_transform_response(args.response_transform_type,training_data),model_predictions,0)
    timers.append(time.time()-start_time)
    write_statistics_to_file(args.output_file,test_error_metrics,training_error_metrics,timers,[training_set_size,test_set_size],model_size,[],[])

if __name__ == "__main__":

    import argparse

    sys.path.insert(0,'%s/../'%(os.getcwd()))
    from arg_defs import add_shared_parameters

    parser = argparse.ArgumentParser()
    add_shared_parameters(parser)
    args, _ = parser.parse_known_args()
    main(args)
