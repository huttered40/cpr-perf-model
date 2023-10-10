def main(args):

    import time,os,sys
    import numpy as np
    #import random as rand
    import pandas as pd

    from cpr_model import cpr_model
    from util import extract_datasets, get_error_metrics,write_statistics_to_file,get_model_size, transform_dataset, transform_predictor, transform_response, inverse_transform_response

    training_df = pd.read_csv(args.training_file, index_col=False, sep=',')
    test_df = pd.read_csv(args.test_file, index_col=False, sep=',')
    param_list = training_df.columns[args.input_columns].tolist()
    data_list = training_df.columns[args.data_columns].tolist()

    # Generate list of model types parameterized on hyper-parameters
    (training_configurations,training_data,training_set_size,\
            test_configurations,test_data,test_set_size,mode_range_min,mode_range_max)\
      = extract_datasets(training_df,test_df,param_list,data_list,args.training_set_size,\
          args.test_set_size,args.mode_range_min,args.mode_range_max)

    if (len(args.parameter_node_spacing_type) != len(param_list)):
        raise AssertionError("Invalid input: parameter_node_spacing_type")
    if (len(args.parameter_node_count) != len(param_list)):
        raise AssertionError("Invalid input: parameter_node_count")

    if (len(args.cp_rank) != 2):
        raise AssertionError("Must specify a CP rank for two tensor models.")

    timers = []
    start_time = time.time()
    model = cpr_model(args.parameter_node_count,args.parameter_type,args.parameter_node_spacing_type,\
                mode_range_min,mode_range_max,args.cp_rank,\
                args.regularization,args.max_spline_degree,\
                args.response_transform_type,args.custom_grid_pts,args.model_convergence_tolerance,args.maximum_num_sweeps,\
                args.factor_matrix_convergence_tolerance,args.maximum_num_iterations,args.barrier_range,\
                args.barrier_reduction_factor,args.projection_set_size_threshold)
    num_tensor_elements,density,loss1,loss2 = model.fit(training_configurations,training_data)
    model_info = [num_tensor_elements,density,loss1,loss2]
    timers.append(time.time()-start_time)

    model_size = get_model_size(model,"Model.joblib")

    model.write_to_file("my_new_data_file_ctf.csv")
    if (args.print_model_parameters):
        model.print_model()

    start_time = time.time()
    model_predictions = []
    for k in range(test_set_size):
        configuration = test_configurations[k,:]*1.
        model_predictions.append(model.predict(configuration))
    test_error_metrics = get_error_metrics(test_set_size,test_configurations,test_data,model_predictions,args.print_test_error)
    timers.append(time.time()-start_time)

    start_time = time.time()
    model_predictions = []
    for k in range(training_set_size):
        configuration = training_configurations[k,:]*1.
        model_predictions.append(model.predict(configuration))
    training_error_metrics = get_error_metrics(training_set_size,training_configurations,training_data,model_predictions,0)
    timers.append(time.time()-start_time)
    write_statistics_to_file(args.output_file,test_error_metrics,training_error_metrics,timers,[training_set_size,test_set_size],model_size,[len(args.parameter_node_count),'-'.join([str(i) for i in args.parameter_node_count]),args.regularization[0],args.cp_rank[0],model_info[0],model_info[1],model_info[2],model_info[3],args.cp_rank[1],args.model_convergence_tolerance[0],args.maximum_num_sweeps,args.factor_matrix_convergence_tolerance,args.maximum_num_iterations,args.barrier_range[0],args.barrier_range[1],args.barrier_reduction_factor,'-'.join([str(i) for i in args.projection_set_size_threshold])],["model:tensor_dim","model:parameter_node_count","model:regularization","model:cp_rank","model:ntensor_elems","model:tensor_density","model:loss1","model:loss2","extrap_cp_rank","model_convergence_tolerance","maximum_num_sweeps","factor_matrix_convergence_tolerance","maximum_num_iterations","barrier_start","barrier_stop","barrier_reduction_factor","projection_set_size_threshold"])

if __name__ == "__main__":

    import argparse
    from arg_defs import add_shared_parameters

    parser = argparse.ArgumentParser()

    add_shared_parameters(parser)
    parser.add_argument(
        '-R',
        '--regularization',
        type=float,
        nargs=2,
        default=[1e-4,1e-4],
        help='Regularization coefficient for both loss functions.')
    parser.add_argument(
        '-c',
        '--cp-rank',
        type=int,
        nargs=2,
        default=[3,1],
        help='Canonical-Polyadic tensor decomposition rank for both models.')
    parser.add_argument(
        '-B',
        '--model-convergence-tolerance',
        type=float,
        nargs=2,
        default=[1e-5,1e-5],
        help='Value of loss function for alternating minimization methods below which optimization of model stops.')
    parser.add_argument(
        '-C',
        '--barrier-range',
        type=float,
        nargs=2,
        default=[1e-11,1e1],
        help='Range of coefficient applied to barrier terms of loss function for initial sweep of Alternating Minimization.')
    parser.add_argument(
        '-H',
        '--maximum-num-sweeps',
        type=int,
        nargs=2,
        default=[100,10],
        help='Maximum number of sweeps of alternating minimization to optimize each model.')
    parser.add_argument(
        '-G',
        '--barrier-reduction-factor',
        type=float,
        default=8,
        help='Value which divides into the coefficient applied to barrier terms for subsequent sweeps of Alternating Minimization.')
    parser.add_argument(
        '-F',
        '--factor-matrix-convergence-tolerance',
        type=float,
        default=1e-3,
        help='Value of relative change in factor matrix update below which optimization of individual factor matrix stops.')
    parser.add_argument(
        '-I',
        '--maximum-num-iterations',
        type=int,
        default=40,
        help='Maximum number of iterations of Newtons method to optimize a factor matrix.')
    parser.add_argument(
        '-K',
        '--parameter-node-count',
        type=int,
        nargs='+',
        required=True,
        help="Number of nodes placed along each parameter's range.")
    parser.add_argument(
        '-O',
        '--parameter-node-spacing-type',
        type=int,
        nargs='+',
        required=True,
        help='Signify the equidistant placement of nodes for each parameter on linear scale (0) or logarithmic scale (1).')
    parser.add_argument(
        '-S',
        '--parameter-type',
        type=int,
        nargs='+',
        required=True,
        help='Signify each parameter as categorical (0), numerical (1), or ordinal (2).')
    parser.add_argument(
        '-J',
        '--max-spline-degree',
        type=int,
        default=3,
        help='Maximum spline degree used for extrapolation model.')
    parser.add_argument(
        '--custom-grid-pts',
        type=float,
        nargs='+',
        default=[],
        help='Node locations for any parameters with parameter-type=2 (specified in corresponding order).')
    parser.add_argument(
        '--projection-set-size-threshold',
        type=int,
        nargs='+',
        default=[],
        help="Minimum number of observations per hyperplane per parameter's corresponding tensor mode from which to construct model.")

    args, _ = parser.parse_known_args()
    main(args)
