# !/usr/bin/python
# coding=utf-8

import os, time, sys, copy
import numpy as np
import numpy.linalg as la
import pandas as pd
import argparse
import arg_defs as arg_defs

import pysgpp as pysgpp
from pysgpp.extensions.datadriven.learner import Types
from pysgpp.extensions.datadriven.learner import LearnerBuilder

sys.path.insert(0,'%s/../'%(os.getcwd()))
from util import extract_datasets, get_error_metrics,write_statistics_to_file,get_model_size,\
                 transform_dataset, transform_predictor, transform_response, inverse_transform_response

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
    model_list = generate_models(nlevels, nadapt_points, reg)

    if (args.print_diagnostics == 1):
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
    #Normalize the data along each mode so max is 1. No log transformation of predictors necessary.
    save_min_values = [0]*len(param_list)
    save_max_values = [0]*len(param_list)
    for i in range(training_configurations.shape[1]):
        save_min_values[i] = (np.amin(training_configurations[:,i]))
        save_max_values[i] = (np.amax(training_configurations[:,i]))
    training_configurations,training_data = transform_dataset([2]*len(param_list),args.response_transform,training_configurations,training_data)
    validation_configurations,validation_data = transform_dataset([0]*len(param_list),0,validation_configurations,validation_data)
    test_configurations,test_data = transform_dataset([0]*len(param_list),0,test_configurations,test_data)
    # Normalize validation and test datasets based on training dataset
    for i in range(validation_configurations.shape[1]):
        validation_configurations[:,i] = validation_configurations[:,i]*1. - save_min_values[i]
        validation_configurations[:,i] = validation_configurations[:,i]*1. / save_max_values[i]
    for i in range(test_configurations.shape[1]):
        test_configurations[:,i] = test_configurations[:,i]*1. - save_min_values[i]
        test_configurations[:,i] = test_configurations[:,i]*1. / save_max_values[i]

    start_time = time.time()
    opt_model_parameters = [-1]*3
    current_best_model = []
    opt_error_metrics = [100000.]*16
    for model_parameters in model_list:
        builder = LearnerBuilder()
        builder.buildRegressor()
        builder.withTrainingDataFromNumPyArray(training_configurations, training_data.copy())
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
        builder.withTestingDataFromNumPyArray(training_configurations,training_data)

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

        results = generate_predictions(learner,validation_configurations)
        model_predictions = []
        for k in range(validation_set_size):
            model_predictions.append(inverse_transform_response(args.response_transform,results[k]))
        validation_error_metrics = get_error_metrics(validation_set_size,validation_configurations,validation_data,model_predictions)
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

    #learner object apparently cannot be dumped. Must ascertain size a different way.
    #model_size = get_model_size(learner.grid,"SGR_Model.joblib")

    start_time = time.time()
    model_predictions = []
    numberGridPoints=Learner.grid.getSize()
    results = generate_predictions(Learner,test_configurations)
    for k in range(test_set_size):
        model_predictions.append(inverse_transform_response(args.response_transform,results[k]))
    timers[2] += (time.time()-start_time)
    test_error_metrics = get_error_metrics(test_set_size,test_configurations,test_data,model_predictions,args.print_test_error)

    model_predictions = []
    results = generate_predictions(Learner,training_configurations)
    for k in range(training_set_size):
        configuration = training_configurations[k,:]*1.
        model_predictions.append(inverse_transform_response(args.response_transform,results[k]))
    training_error_metrics = get_error_metrics(training_set_size,training_configurations,inverse_transform_response(args.response_transform,training_data),model_predictions,0)
    write_statistics_to_file(args.output_file,test_error_metrics,training_error_metrics,timers,[training_set_size,validation_set_size,test_set_size],numberGridPoints*(training_configurations.shape[1]*4+8),[opt_model_parameters[0],opt_model_parameters[1],opt_model_parameters[2],args.nrefinements,numberGridPoints,numberGridPoints*(training_configurations.shape[1]*4+8)],["model:nlevels","model:nadaptpts","model:reg","model:nrefinements","model:NumberGridPoints","model:analytic_model_size"])    
