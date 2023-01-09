# !/usr/bin/python
# coding=utf-8

import os, joblib, time, sys, copy
import numpy as np
import numpy.linalg as la
import pandas as pd
import argparse
import arg_defs as arg_defs

import pysgpp as pysgpp
from pysgpp.extensions.datadriven.learner import Types
from pysgpp.extensions.datadriven.learner import LearnerBuilder

sys.path.insert(0,'%s/../'%(os.getcwd()))
from util import extract_datasets, get_error_metrics

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

    timers = [0.]*3
    training_df = pd.read_csv('%s'%(args.training_file), index_col=0, sep=',')
    test_df = pd.read_csv('%s'%(args.test_file), index_col=0, sep=',')
    param_list = training_df.columns[[int(n) for n in args.input_columns.split(',')]].tolist()
    data_list = training_df.columns[[int(n) for n in args.data_columns.split(',')]].tolist()

    nlevels = [int(n) for n in args.nlevels.split(',')]
    nadapt_points = [int(n) for n in args.nadaptpts.split(',')]
    reg = [float(n) for n in args.reg.split(',')]
    # Note: no option to transform data because basis functions assume certain structure,
    #   and transforming the runtimes would necessitate transforming the basis functions.

    # Generate list of model types parameterized on hyper-parameters
    model_list = generate_models(nlevels, nadapt_points, reg)

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
        _test_data_ = test_data.copy()
        if (args.response_transform == 1):
            _test_data_ = np.log(_test_data_)
        builder.withTestingDataFromNumPyArray(test_inputs, _test_data_)

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
        gridStorage = learner.grid.getStorage()
        """
        print("what are these sizes - ", numberGridPoints,sys.getsizeof(learner),sys.getsizeof(learner.grid))
        print("What is this - ", gridStorage.getDimension(), gridStorage.getSize(), gridStorage.getMaxLevel(), gridStorage.getNumberOfInnerPoints())
        print("What is this size - ", sys.getsizeof(gridStorage))
        print(gridStorage)
        print(gridStorage.toString())
        for ii in range(gridStorage.getSize()):
            print("check - ", *gridStorage.getPoint(ii))
        """

        # Use validation set again here.
        results = generate_predictions(learner,test_inputs)
        # Now validate on validation set
        model_predictions = []
        for k in range(validation_set_size):
            _data_ = results[k]
            if (args.response_transform == 1):
                _data_ = np.exp(_data_)
            model_predictions.append(_data_)

        validation_error_metrics = get_error_metrics(validation_set_size,validation_inputs,validation_data,model_predictions)
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

    """ learner object apparently cannot be dumped. Must ascertain size a different way.
    joblib.dump(learner.grid, "SGR_Model.joblib") 
    model_size = os.path.getsize('SGR_Model.joblib')
    print("SGR model size: %f bytes"%(model_size))
    """

    start_time = time.time()
    model_predictions = []
    numberGridPoints=Learner.grid.getSize()
    results = generate_predictions(Learner,test_inputs)
    for k in range(test_set_size):
        _data_ = results[k]
        if (args.response_transform == 1):
            _data_ = np.exp(_data_)
        model_predictions.append(_data_)
    timers[2] += (time.time()-start_time)

    test_error_metrics = get_error_metrics(test_set_size,test_inputs,test_data,model_predictions)

    # Write relevant error statistics to file
    columns = (\
        "input:training_set_size",\
        "input:test_set_size",\
        "input:response_transform",\
        "nlevels",\
        "nadaptpts",\
        "reg",\
        "nrefinements",\
        "NumberGridPoints",\
        "analytic_model_size",\
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
        columns[6] : args.nrefinements,\
        columns[7] : numberGridPoints,\
        columns[8] : numberGridPoints*(training_inputs.shape[1]*4+8),\
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
