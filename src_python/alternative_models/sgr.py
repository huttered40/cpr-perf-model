import os, sys
import pysgpp as pysgpp

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

def main(args):

    import time
    import numpy as np
    import pandas as pd

    from pysgpp.extensions.datadriven.learner import Types
    from pysgpp.extensions.datadriven.learner import LearnerBuilder

    sys.path.insert(0,'%s/../'%(os.getcwd()))
    from util import extract_datasets, get_error_metrics,write_statistics_to_file,get_model_size, transform_dataset, transform_predictor, transform_response, inverse_transform_response

    timers = [0.]*3
    training_df = pd.read_csv('%s'%(args.training_file), index_col=0, sep=',')
    test_df = pd.read_csv('%s'%(args.test_file), index_col=0, sep=',')
    param_list = training_df.columns[args.input_columns].tolist()
    data_list = training_df.columns[args.data_columns].tolist()

    (training_configurations,training_data,training_set_size,\
            test_configurations,test_data,test_set_size,mode_range_min,mode_range_max)\
      = extract_datasets(training_df,test_df,param_list,data_list,args.training_set_size,\
          args.test_set_size,args.mode_range_min,args.mode_range_max)
    #Normalize the data along each mode so max is 1. No log transformation of predictors necessary.
    save_min_values = [0]*len(param_list)
    save_max_values = [0]*len(param_list)
    for i in range(training_configurations.shape[1]):
        save_min_values[i] = (np.amin(training_configurations[:,i]))
        save_max_values[i] = (np.amax(training_configurations[:,i]))
    training_configurations,training_data = transform_dataset([2]*len(param_list),args.response_transform_type,training_configurations,training_data)
    test_configurations,test_data = transform_dataset([0]*len(param_list),0,test_configurations,test_data)
    for i in range(test_configurations.shape[1]):
        test_configurations[:,i] = test_configurations[:,i]*1. - save_min_values[i]
        test_configurations[:,i] = test_configurations[:,i]*1. / save_max_values[i]

    start_time = time.time()
    builder = LearnerBuilder()
    builder.buildRegressor()
    builder.withTrainingDataFromNumPyArray(training_configurations, training_data.copy())
    builder = builder.withGrid().withBorder(Types.BorderTypes.NONE)
    builder.withLevel(args.nlevels)
    # add specification descriptor
    builder = builder.withSpecification()
    # in spec. descriptor, set lambda value to 1e-6 and
    # use k grid-points that shall be refined during refinement step
    builder.withLambda(args.regularization).withAdaptPoints(args.nadaptpts)
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

    timers[1] += (time.time()-start_time)

    #learner object apparently cannot be dumped. Must ascertain size a different way.
    #model_size = get_model_size(learner.grid,"SGR_Model.joblib")

    start_time = time.time()
    model_predictions = []
    numberGridPoints=learner.grid.getSize()
    results = generate_predictions(learner,test_configurations)
    for k in range(test_set_size):
        model_predictions.append(inverse_transform_response(args.response_transform_type,results[k]))
    timers[2] += (time.time()-start_time)
    test_error_metrics = get_error_metrics(test_set_size,test_configurations,test_data,model_predictions,args.print_test_error)

    model_predictions = []
    results = generate_predictions(learner,training_configurations)
    for k in range(training_set_size):
        configuration = training_configurations[k,:]*1.
        model_predictions.append(inverse_transform_response(args.response_transform_type,results[k]))
    training_error_metrics = get_error_metrics(training_set_size,training_configurations,inverse_transform_response(args.response_transform_type,training_data),model_predictions,0)
    write_statistics_to_file(args.output_file,test_error_metrics,training_error_metrics,timers,[training_set_size,test_set_size],numberGridPoints*(training_configurations.shape[1]*4+8),[args.nlevels,args.nadaptpts,args.regularization,args.nrefinements,numberGridPoints,numberGridPoints*(training_configurations.shape[1]*4+8)],["model:nlevels","model:nadaptpts","model:reg","model:nrefinements","model:NumberGridPoints","model:analytic_model_size"])    

if __name__ == "__main__":

    import argparse

    sys.path.insert(0,'%s/../'%(os.getcwd()))
    from arg_defs import add_shared_parameters

    parser = argparse.ArgumentParser()
    add_shared_parameters(parser)
    parser.add_argument(
        '--nlevels',
        type=int,
        default=3,
        help='Number of sparse grid levels.')
    parser.add_argument(
        '--nadaptpts',
        type=int,
        default=3,
        help='Number of grid-points to update.')
    parser.add_argument(
        '-R',
        '--regularization',
        type=float,
        default=1e-4,
        help='Regularization coefficient(s) provided to loss function.')
    parser.add_argument(
        '--nrefinements',
        type=int,
        default=5,
        help='Number of sparse-grid refinements.')
    args, _ = parser.parse_known_args()
    main(args)
