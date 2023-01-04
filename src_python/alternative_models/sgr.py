# !/usr/bin/python
# coding=utf-8

import time, sys, copy
import numpy as np
import numpy.linalg as la
import scipy.stats as scst
import random as rand
import os
import pandas as pd
import argparse
import arg_defs as arg_defs
import pysgpp as pysgpp
from pysgpp.extensions.datadriven.learner import Types
from pysgpp.extensions.datadriven.learner import LearnerBuilder

def generate_predictions(learner,features):
    # create SGPP DataMatrix with point data; each
    # row of the matrix contains one line of parameters from validation[features]
    points=pysgpp.DataMatrix(features.shape[0],features.shape[1])
    for i in range(features.shape[0]):
        tmpVec = pysgpp.DataVector(features.shape[1])
        for j in range(features.shape[1]):
            tmpVec[j] = features[i,j]
        points.setRow(i,tmpVec)
    # compute the sparse grid solution on these points
    result=learner.applyData(points)
    return result

def mse_validation(learner,data,features):
    result=generate_predictions(leader,features)
    
    # compute and return mean squared error
    mse=0.0
    myerr=0.0
    myrelerr=0.0
    meanerr=0.0
    meanrelerr=0.0
    for i in range(features.shape[0]):
        meanerr=meanerr + abs(result[i]-data[i])
        meanrelerr=meanrelerr + abs(result[i]-data[i])/abs(data[i])
        mse = mse + (result[i]-data[i])**2
        if abs(result[i]-data[i])>myerr:
            myerr=abs(result[i]-data[i])
        if abs(result[i]-data[i])/abs(data[i])>myrelerr:
            myrelerr=abs(result[i]-data[i])/abs(data[i])
    return mse/data.shape[0],meanerr/data.shape[0],meanrelerr/data.shape[0],myerr,myrelerr

def generate_models(_nlevels,_nadapt_points,_reg):
    model_list = []
    for i in _nlevels:
        for j in _nadapt_points:
            for k in _reg:
                model_list.append([i,j,k])
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

    nlevels = [int(n) for n in args.nlevels.split(',')]
    nadapt_points = [int(n) for n in args.nadaptpts.split(',')]
    reg = [float(n) for n in args.reg.split(',')]
    # Note: no option to transform data because basis functions assume certain structure,
    #   and transforming the runtimes would necessitate transforming the basis functions.

    # Generate list of model types parameterized on hyper-parameters
    model_list = generate_models(nlevels, nadapt_points, reg)
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

    # we check the refined grids for the number of total grid points. If the number of grid points
    # exceeds 0.2*training_set_size (i.e., we use more grid points than 20% of the number of samples),
    # we will not refine the grid further and use the corresponding result.
    # We will always use the first grid constellation, even if this one does not satisfy this constraint.
    factor_max_points=1.0

    #Normalize the data along each mode so max is 1.
    for i in range(training_inputs.shape[1]):
        training_inputs[:,i] -= (np.amin(training_inputs[:,i]))
        training_inputs[:,i] = training_inputs[:,i]*1. / np.amax(training_inputs[:,i])
    for i in range(test_inputs.shape[1]):
        test_inputs[:,i] -= (np.amin(test_inputs[:,i]))
        test_inputs[:,i] = test_inputs[:,i]*1. / np.amax(test_inputs[:,i])

    start_time = time.time()
    opt_model_parameters = [-1]*3
    current_best_model = []
    opt_error_metrics = [100000.]*6    # arithmetic sum of log Q, arithmetic sum of log^2 Q, geometric mean of relative errors, MAPE, SMAPE, RMSE
    for model_parameters in model_list:
        start_time_generate = time.time()
        builder = LearnerBuilder()
        builder.buildRegressor()
        _training_data_ = training_data.copy()
        if (args.response_transform == 1):
            _training_data_ = np.log(_training_data_)
        #print("what is this range - ", np.amax(_training_data_) / np.amin(_training_data_))
        builder.withTrainingDataFromNumPyArray(training_inputs, _training_data_)
        builder = builder.withGrid().withBorder(Types.BorderTypes.NONE)
        builder.withLevel(model_parameters[0])
        # add specification descriptor
        builder = builder.withSpecification()
        # in spec. descriptor, set lambda value to 1e-6 and
        # use k grid-points that shall be refined during refinement step
        builder.withLambda(model_parameters[2]).withAdaptPoints(model_parameters[1])
        # use identity operator according to Eq. (6.5) in D. Pflüger, PhD thesis, 2010
        builder.withIdentityOperator()
        # use stop policy and a CG solver with an accuracy of 1e-4, or a maximum of 1000 iterations
        builder = builder.withStopPolicy()
        builder = builder.withCGSolver()
        builder.withAccuracy(0.0001)
        builder.withImax(100)    # this impacts accuracy and number of grid-points!
        _test_data_ = test_data[:split_idx].copy()
        if (args.response_transform == 1):
            _test_data_ = np.log(_test_data_)
        builder.withTestingDataFromNumPyArray(test_inputs[:split_idx,:], _test_data_)

        start_time_solve = time.time()
        # Extract the corresponding learner and learn, using the test data to prevent overfitting
        learner = builder.andGetResult()
        learner.learnDataWithTest()
        timers[0] += (time.time()-start_time_solve)

        # do not perform any kind of refinement or writing to file, if the default grid already yields
        # potentially overfitting; this grid will be, nevertheless, saved
        counter_refinements=0
        # iterate as long as we do not have too much grid points compared to number of samples
        # and we do not exceed the maximum number of refinement steps
        while counter_refinements<args.nrefinements:
            # do refinement
            start_time_solve = time.time()
            learner.refineGrid()
            learner.learnDataWithTest()
            timers[0] += (time.time()-start_time_solve)
            counter_refinements=counter_refinements+1
        numberGridPoints=learner.grid.getSize()
        #print("what are these sizes - ", numberGridPoints,sys.getsizeof(learner),sys.getsizeof(learner.grid))
        gridStorage = learner.grid.getStorage()
        #print("What is this - ", gridStorage.getDimension(), gridStorage.getSize(), gridStorage.getMaxLevel(), gridStorage.getNumberOfInnerPoints())
        #print("What is this size - ", sys.getsizeof(gridStorage))
        #print(gridStorage)
        #print(gridStorage.toString())
        """
        for ii in range(gridStorage.getSize()):
            print("check - ", *gridStorage.getPoint(ii))
        """

        # Use validation set again here.
        results = generate_predictions(learner,test_inputs[:split_idx,:])
        # Now validate on validation set
        model_predictions = []
        for k in range(validation_set_size):
            _data_ = results[k]
            if (args.response_transform == 1):
                _data_ = np.exp(_data_)
            model_predictions.append(_data_)

        validation_error_metrics = [0]*10
        prediction_errors = [[] for k in range(3)]
        for k in range(validation_set_size):
            prediction_errors[0].append(np.log((model_predictions[k] if model_predictions[k] > 0 else 1e-14)/test_data[k]))
            prediction_errors[1].append(np.abs(model_predictions[k]-test_data[k])/test_data[k])
            if (prediction_errors[1][-1] <= 0):
                prediction_errors[1][-1] = 1e-14
            prediction_errors[2].append(np.abs(model_predictions[k]-test_data[k])/np.average([model_predictions[k],test_data[k]]))
        if (validation_set_size>0):
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
            print("Validation Error for (nlevels=%d,nadaptpts=%d,reg=%f,nrefinements=%d) is "%(model_parameters[0],model_parameters[1],model_parameters[2], counter_refinements),validation_error_metrics, "with runtime ", timers)
            if (validation_error_metrics[2] < opt_error_metrics[1]):
                opt_model_parameters = copy.deepcopy(model_parameters)
                opt_error_metrics = copy.deepcopy(validation_error_metrics)
                current_best_model = learner
    if (validation_set_size>0):
        Learner = current_best_model
    else:
        Learner = learner
        opt_model_parameters = model_parameters
    timers[1] += (time.time()-start_time)

    start_time = time.time()
    model_predictions = []
    numberGridPoints=Learner.grid.getSize()
    results = generate_predictions(Learner,test_inputs[split_idx:,:])
    for k in range(test_set_size):
        _data_ = results[k]
        if (args.response_transform == 1):
            _data_ = np.exp(_data_)
        model_predictions.append(_data_)
    timers[2] += (time.time()-start_time)

    test_error_metrics = [0]*12
    prediction_errors = [[] for k in range(3)]
    for k in range(test_set_size):
        prediction_errors[0].append(np.log((model_predictions[k] if model_predictions[k] > 0 else 1e-14)/test_data[split_idx+k]))
        prediction_errors[1].append(np.abs(model_predictions[k]-test_data[split_idx+k])/test_data[split_idx+k])
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
    for k in range(test_set_size):
        print(np.absolute(prediction_errors[0][k]))
    """

    columns = (\
        "input:training_set_size",\
        "input:test_set_size",\
        "input:response_transform",\
        "nlevels",\
        "nadaptpts",\
        "reg",\
        "nrefinements",\
        "NumberGridPoints",\
        "Model_size",\
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
        columns[2] : args.response_transform,\
        columns[3] : opt_model_parameters[0],\
        columns[4] : opt_model_parameters[1],\
        columns[5] : opt_model_parameters[2],\
        columns[6] : numberGridPoints,\
        columns[7] : numberGridPoints*(training_inputs.shape[1]*4+8),\
        columns[8] : args.nrefinements,\
    columns[9] : test_error_metrics[2],\
    columns[10] : test_error_metrics[4],\
    columns[11] : test_error_metrics[6],\
    columns[12] : test_error_metrics[8],\
    columns[13] : test_error_metrics[10],\
        columns[14] : timers[0],\
        columns[15] : timers[1],\
        columns[16] : timers[2],\
    } }
    test_results_df = pd.DataFrame(data=test_results_dict,index=columns).T
    test_results_df.to_csv("%s"%(args.output_file),sep=',',header=1,mode="a")
