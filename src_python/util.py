import numpy as np
import random as rand
import scipy.stats as scst
import pandas as pd

def extract_datasets(training_df,test_df,param_list,data_list,training_set_size,\
      test_set_size,split_percentage,user_specified_mode_range_min,user_specified_mode_range_max):
    # NOTE: assumption that training/test input files follow same format
    # Randomize test set
    np.random.seed(10)
    x_test = np.array(range(len(test_df[param_list].values)))
    np.random.shuffle(x_test)
    test_inputs = test_df[param_list].values[x_test]
    test_data = test_df[data_list].values.reshape(-1)[x_test]

    # Randomize training set
    x_train = np.array(range(len(training_df[param_list].values)))
    np.random.shuffle(x_train)
    training_inputs = training_df[param_list].values[x_train]
    training_data = training_df[data_list].values.reshape(-1)[x_train]

    # Final selection of test set
    test_set_size = min(test_set_size,test_inputs.shape[0])
    test_inputs = test_inputs[:test_set_size,:]
    test_data = test_data[:test_set_size]

    # Split training data
    split_idx = int(split_percentage * training_set_size)
    training_set_size = min(training_inputs.shape[0]-split_idx,training_set_size)
    validation_set_size = split_idx

    # Final selection of training set
    training_inputs = training_inputs[split_idx:(training_set_size+split_idx),:]
    training_data = training_data[split_idx:(training_set_size+split_idx)]

    # Final selection of validation set
    validation_inputs = training_inputs[:split_idx]
    validation_data = training_data[:split_idx]

    #print("Training set size: %d"%(training_set_size))
    #print("Validation set size: %d"%(validation_set_size))
    #print("Test set size: %d"%(test_set_size))
    #print("training_inputs: ", training_inputs)
    #print("training_data: ", training_data)
    #print("test_inputs: ", test_inputs)
    #print("test_data: ", test_data)

    test_inputs = test_inputs.astype(np.float64)
    training_inputs = training_inputs.astype(np.float64)
    mode_range_min = [0]*len(param_list)
    mode_range_max = [0]*len(param_list)
    if (user_specified_mode_range_min == '' or user_specified_mode_range_max == ''):
        for i in range(training_inputs.shape[1]):
            mode_range_min[i] = np.amin(training_inputs[:,i])
            mode_range_max[i] = np.amax(training_inputs[:,i])
    else:
        mode_range_min = [float(n) for n in user_specified_mode_range_min.split(',')]
        mode_range_max = [float(n) for n in user_specified_mode_range_max.split(',')]
    #print("mode_range_min: ",mode_range_min)
    #print("mode_range_max: ",mode_range_max)
    assert(len(mode_range_min)==len(param_list))
    assert(len(mode_range_max)==len(param_list))

    return (training_inputs,training_data,training_set_size,\
            validation_inputs,validation_data,validation_set_size,\
            test_inputs,test_data,test_set_size,mode_range_min,mode_range_max\
           )

def get_error_metrics(set_size,inputs,data,model_predictions):
    error_metrics = [0]*12
    if (set_size == 0):
        return error_metrics
    prediction_errors = [[] for k in range(3)]        # metrics 0 and 1 borrow, metrics 2 and 3 borrow, and rsme has already been calculated, so can reduce from 6 to 3
    for k in range(set_size):
        input_tuple = inputs[k,:]*1.        # NOTE: without this cast from into to float, interpolation produces zeros
        prediction_errors[0].append(np.log((model_predictions[k] if model_predictions[k] > 0 else 1e-14)/data[k]))
        prediction_errors[1].append(np.abs(model_predictions[k]-data[k])/data[k])
        if (prediction_errors[1][-1] <= 0):
            prediction_errors[1][-1] = 1e-14
        prediction_errors[2].append(np.abs(model_predictions[k]-data[k])/np.average([model_predictions[k],data[k]]))
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
    """
    for k in range(len(prediction_errors[0])):
        print(np.absolute(prediction_errors[0][k]))
    """
    return error_metrics
