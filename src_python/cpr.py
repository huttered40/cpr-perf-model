def main(args):

    import time,os,sys
    import numpy as np
    import random as rand
    import pandas as pd

    from cpr_model import cpr_model
    from util import extract_datasets, get_error_metrics,write_statistics_to_file,get_model_size, transform_dataset, transform_predictor, transform_response, inverse_transform_response

    timers = [0.]*3
    training_df = pd.read_csv('%s'%(args.training_file), index_col=0, sep=',')
    test_df = pd.read_csv('%s'%(args.test_file), index_col=0, sep=',')
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

    start_time = time.time()
    start_time_solve = time.time()
    cpr_mod = cpr_model(args.parameter_node_count,args.parameter_type,args.parameter_node_spacing_type,\
                mode_range_min,mode_range_max,args.cp_rank,args.cp_rank_for_extrapolation_model,args.loss_function,\
                args.regularization,args.max_spline_degree,\
                args.response_transform_type,args.custom_grid_pts,args.model_convergence_tolerance,args.maximum_num_sweeps,\
                args.factor_matrix_convergence_tolerance,args.maximum_num_iterations,args.barrier_start,args.barrier_stop,\
                args.barrier_reduction_factor,args.projection_set_size_threshold,args.build_extrapolation_model)
    print(training_configurations)
    print(training_data)
    num_tensor_elements,density,loss1,loss2 = cpr_mod.fit(training_configurations,training_data)
    timers[0] += (time.time()-start_time_solve)
    model_info = [num_tensor_elements,density,loss1,loss2]
    timers[1] += (time.time()-start_time)

    model_size = get_model_size(cpr_mod,"CPR_Model.joblib")

    if (args.print_model_parameters):
        cpr_mod.print_parameters()

    start_time = time.time()
    model_predictions = []
    for k in range(test_set_size):
        configuration = test_configurations[k,:]*1.
        model_predictions.append(cpr_mod.predict(configuration))
    timers[2] += (time.time()-start_time)
    test_error_metrics = get_error_metrics(test_set_size,test_configurations,test_data,model_predictions,args.print_test_error)
    model_predictions = []
    for k in range(training_set_size):
        configuration = training_configurations[k,:]*1.
        model_predictions.append(cpr_mod.predict(configuration))
    training_error_metrics = get_error_metrics(training_set_size,training_configurations,training_data,model_predictions,0)
    write_statistics_to_file(args.output_file,test_error_metrics,training_error_metrics,timers,[training_set_size,test_set_size],model_size,[len(args.parameter_node_count),'-'.join([str(i) for i in args.parameter_node_count]),args.regularization,args.cp_rank,model_info[0],model_info[1],model_info[2],model_info[3],args.cp_rank_for_extrapolation_model,args.loss_function,args.model_convergence_tolerance,args.maximum_num_sweeps,args.factor_matrix_convergence_tolerance,args.maximum_num_iterations,args.barrier_start,args.barrier_stop,args.barrier_reduction_factor,'-'.join([str(i) for i in args.projection_set_size_threshold])],["model:tensor_dim","model:parameter_node_count","model:regularization","model:cprank","model:ntensor_elems","model:tensor_density","model:loss1","model:loss2","extrap_cp_rank","loss_function","model_convergence_tolerance","maximum_num_sweeps","factor_matrix_convergence_tolerance","maximum_num_iterations","barrier_start","barrier_stop","barrier_reduction_factor","projection_set_size_threshold"])

if __name__ == "__main__":

    import argparse
    from arg_defs import add_shared_parameters

    parser = argparse.ArgumentParser()

    add_shared_parameters(parser)
    parser.add_argument(
        '-E',
        '--build-extrapolation-model',
        action='store_true',
        help='Enable extrapolation of program execution time beyond provided input parameters.')
    parser.add_argument(
        '-R',
        '--regularization',
        type=float,
        default=1e-4,
        help='Regularization coefficient(s) provided to loss function.')
    parser.add_argument(
        '-c',
        '--cp-rank',
        type=int,
        default=3,
        help='Canonical-Polyadic tensor decomposition rank.')
    parser.add_argument(
        '-B',
        '--model-convergence-tolerance',
        type=float,
        default=1e-5,
        help='Value of loss function for alternating minimization methods below which optimization of model stops.')
    parser.add_argument(
        '-F',
        '--factor-matrix-convergence-tolerance',
        type=float,
        default=1e-3,
        help='Value of relative change in factor matrix update below which optimization of individual factor matrix stops.')
    parser.add_argument(
        '-C',
        '--barrier-start',
        type=float,
        default=1e1,
        help='Initial value of coefficient applied to barrier terms of loss function for initial sweep of Alternating Minimization.')
    parser.add_argument(
        '-D',
        '--barrier-stop',
        type=float,
        default=1e-11,
        help='Minimum value of coefficient applied to barrier terms for initial sweep of Alternating Minimization.')
    parser.add_argument(
        '-G',
        '--barrier-reduction-factor',
        type=float,
        default=8,
        help='Value which divides into the coefficient applied to barrier terms for subsequent sweeps of Alternating Minimization.')
    parser.add_argument(
        '-H',
        '--maximum-num-sweeps',
        type=int,
        default=100,
        help='Maximum number of sweeps of alternating minimization to optimize model.')
    parser.add_argument(
        '-I',
        '--maximum-num-iterations',
        type=int,
        default=40,
        help='Maximum number of iterations of Newtons method to optimize a factor matrix.')
    parser.add_argument(
        '-Q',
        '--cp-rank-for-extrapolation-model',
        type=int,
        default=1,
        help='Canonical-Polyadic tensor decomposition rank for .. do this depending on loss, not for .')
    parser.add_argument(
        '-L',
        '--loss-function',
        type=int,
        default=0,
        help='Loss function to optimize model using provided dataset.')
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
