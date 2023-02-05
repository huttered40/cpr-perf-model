import numpy as np
import random as rand
import scipy.stats as scst
import pandas as pd
import os,joblib

def extract_datasets(training_df,test_df,param_list,data_list,training_set_size,\
      test_set_size,split_percentage,user_specified_mode_range_min,user_specified_mode_range_max,print_diagnostics=0):
    # NOTE: assumption that training/test files follow same format
    # Randomize test set
    np.random.seed(10)
    x_test = np.array(range(len(test_df[param_list].values)))
    np.random.shuffle(x_test)
    test_configurations = test_df[param_list].values[x_test]
    test_data = test_df[data_list].values.reshape(-1)[x_test]

    # Randomize training set
    x_train = np.array(range(len(training_df[param_list].values)))
    np.random.shuffle(x_train)
    training_configurations = training_df[param_list].values[x_train]
    training_data = training_df[data_list].values.reshape(-1)[x_train]

    # Final selection of test set
    test_set_size = min(test_set_size,test_configurations.shape[0])
    test_configurations = test_configurations[:test_set_size,:]
    test_data = test_data[:test_set_size]

    # Split training data
    split_idx = int(split_percentage * training_set_size)
    training_set_size = min(training_configurations.shape[0]-split_idx,training_set_size)
    validation_set_size = split_idx

    # Final selection of training set
    training_configurations = training_configurations[split_idx:(training_set_size+split_idx),:]
    training_data = training_data[split_idx:(training_set_size+split_idx)]

    # Find maximum and minimum training execution times
    min_time = np.amin(training_data)
    max_time = np.amax(training_data)
    print("Min time - ", min_time)
    print("Max time - ", max_time)

    # Final selection of validation set
    validation_configurations = training_configurations[:split_idx]
    validation_data = training_data[:split_idx]

    if (print_diagnostics == 1):
        print("Training set size: %d"%(training_set_size))
        print("Validation set size: %d"%(validation_set_size))
        print("Test set size: %d"%(test_set_size))
        print("training_configurations: ", training_configurations)
        print("training_data: ", training_data)
        print("test_configurations: ", test_configurations)
        print("test_data: ", test_data)

    test_configurations = test_configurations.astype(np.float64)
    training_configurations = training_configurations.astype(np.float64)
    mode_range_min = [0]*len(param_list)
    mode_range_max = [0]*len(param_list)
    if (user_specified_mode_range_min == '' or user_specified_mode_range_max == ''):
        for i in range(training_configurations.shape[1]):
            mode_range_min[i] = np.amin(training_configurations[:,i])
            mode_range_max[i] = np.amax(training_configurations[:,i])
    else:
        mode_range_min = [float(n) for n in user_specified_mode_range_min.split(',')]
        mode_range_max = [float(n) for n in user_specified_mode_range_max.split(',')]
    if (print_diagnostics == 1):
        print("mode_range_min: ",mode_range_min)
        print("mode_range_max: ",mode_range_max)
    assert(len(mode_range_min)==len(param_list))
    assert(len(mode_range_max)==len(param_list))

    return (training_configurations,training_data,training_set_size,\
            validation_configurations,validation_data,validation_set_size,\
            test_configurations,test_data,test_set_size,mode_range_min,mode_range_max\
           )

"""
NOTE: Eight aggregate error metrics are calculated (both mean and standard deviation):
      0. arithmetic sum of log(accuracy ratio)
      1. arithmetic sum of abs(log(accuracy ratio))
      2. arithmetic sum of log(accuracy ratio)^2
      3. geometric mean of relative error
      4. mean absolute percentage error
      5. symmetric mean absolute percentage error
      6. mean squared error
      7. mean absolute error
"""
def get_error_metrics(set_size,configurations,data,model_predictions,print_errors=0):
    error_metrics = [0]*16
    if (set_size == 0):
        return error_metrics
    prediction_errors = [[] for k in range(5)]        # metrics 0 and 1 borrow, metrics 2 and 3 borrow, and rsme has already been calculated, so can reduce from 6 to 3
    for k in range(set_size):
        input_tuple = configurations[k,:]*1.        # NOTE: without this cast from into to float, interpolation produces zeros
        prediction_errors[0].append(np.log((model_predictions[k] if model_predictions[k] > 0 else 1e-14)/data[k]))
        prediction_errors[1].append(np.absolute(model_predictions[k]-data[k])/data[k])
        prediction_errors[2].append(np.absolute(model_predictions[k]-data[k])/np.average([model_predictions[k],data[k]]))
        prediction_errors[3].append((model_predictions[k]-data[k])**2)
        prediction_errors[4].append(np.absolute(model_predictions[k]-data[k]))
    error_metrics[0] = np.average(prediction_errors[0])
    error_metrics[1] = np.std(prediction_errors[0],ddof=1)
    error_metrics[2] = np.average(np.absolute(prediction_errors[0]))
    error_metrics[3] = np.std(np.absolute(prediction_errors[0]),ddof=1)
    error_metrics[4] = np.average(np.asarray(prediction_errors[0])**2)
    error_metrics[5] = np.std(np.asarray(prediction_errors[0])**2,ddof=1)
    error_metrics[6] = scst.gmean(prediction_errors[1])
    error_metrics[7] = np.exp(np.std(np.log(prediction_errors[1]),ddof=1))
    error_metrics[8] = np.average(prediction_errors[1])
    error_metrics[9] = np.std(prediction_errors[1],ddof=1)
    error_metrics[10] = np.average(prediction_errors[2])
    error_metrics[11] = np.std(prediction_errors[2],ddof=1)
    error_metrics[12] = np.average(prediction_errors[3])
    error_metrics[13] = np.std(prediction_errors[3],ddof=1)
    error_metrics[14] = np.average(prediction_errors[4])
    error_metrics[15] = np.std(prediction_errors[4],ddof=1)
    if (print_errors):
        for k in range(len(prediction_errors[0])):
            print("%d"%(k)),
            for kk in range(len(configurations[k,:])):
                print("%f"%(configurations[k,kk])),
            print("%f,%f,%f,%f"%(data[k],model_predictions[k],np.absolute(prediction_errors[0][k]),prediction_errors[0][k]**2))
    return error_metrics

"""
NOTE: Other transformations, including cost-based transformations and box-cox transformations were investigated.
      However, log transformations proved most robust, which is also supported in previous literature.
"""
def transform_dataset(predictor_transform,response_transform,configurations,data):
    if (response_transform==1):
        data = np.log(data)
    for i in range(configurations.shape[1]):
        if (predictor_transform[i]==1):
            configurations[:,i] = np.log(configurations[:,i])
        elif (predictor_transform[i]==2):
            configurations[:,i] = configurations[:,i]*1. - (np.amin(configurations[:,i]))
            configurations[:,i] = configurations[:,i]*1. / np.amax(configurations[:,i])
    return (configurations,data)

def transform_predictor(predictor_transform,configuration):
    for i in range(len(configuration)):
        if (predictor_transform[i]==1):
            configuration[i] = np.log(configuration[i])
        elif (predictor_transform[i]==2):
            assert(0)# No need for this, should be done as an entire dataset
    return configuration

def transform_response(response_transform,data):
    if (response_transform==1):
        data = np.log(data)
    return data

def inverse_transform_response(response_transform,data):
    if (response_transform==1):
        data = np.exp(1)**(data)
    return data

def get_model_size(model,model_str):
    joblib.dump(model,model_str) 
    model_size = os.path.getsize(model_str)
    return model_size

def write_statistics_to_file(output_file,test_error_summary_statistics,training_error_summary_statistics,timers,inputs,model_size,model_info,model_info_strings):
    columns = [\
         "training_error:mlogq",\
         "training_error:mabslogq",\
        "training_error:mlogq2",\
        "training_error:gmre",\
        "training_error:mape",\
        "training_error:smape",\
        "training_error:mse",\
        "training_error:mae",\
         "test_error:mlogq",\
         "test_error:mabslogq",\
        "test_error:mlogq2",\
        "test_error:gmre",\
        "test_error:mape",\
        "test_error:smape",\
        "test_error:mse",\
        "test_error:mae",\
        "time:model_fit",\
        "time:model_configuration",\
        "time:model_eval",\
        "input:training_set_size",\
        "input:validation_set_size",\
        "input:test_set_size",\
        "model:size"]
    columns += model_info_strings
    write_dict = {0:{\
        columns[0] : training_error_summary_statistics[0],\
        columns[1] : training_error_summary_statistics[2],\
        columns[2] : training_error_summary_statistics[4],\
        columns[3] : training_error_summary_statistics[6],\
        columns[4] : training_error_summary_statistics[8],\
        columns[5] : training_error_summary_statistics[10],\
        columns[6] : training_error_summary_statistics[12],\
        columns[7] : training_error_summary_statistics[14],\
        columns[8] : test_error_summary_statistics[0],\
        columns[9] : test_error_summary_statistics[2],\
        columns[10] : test_error_summary_statistics[4],\
        columns[11] : test_error_summary_statistics[6],\
        columns[12] : test_error_summary_statistics[8],\
        columns[13] : test_error_summary_statistics[10],\
        columns[14] : test_error_summary_statistics[12],\
        columns[15] : test_error_summary_statistics[14],\
        columns[16] : timers[0],\
        columns[17] : timers[1],\
        columns[18] : timers[2],\
        columns[19] : inputs[0],\
        columns[20] : inputs[1],\
        columns[21] : inputs[2],\
        columns[22] : model_size,\
    }}
    for i in range(len(model_info)):
        write_dict[0][columns[23+i]] = model_info[i]
    results_df = pd.DataFrame(data=write_dict,index=columns).T
    results_df.to_csv("%s"%(output_file),sep=',',header=1,mode="a")
