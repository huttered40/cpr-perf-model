import time,os,sys,joblib,copy,argparse
import numpy as np
import random as rand
import pandas as pd
import arg_defs as arg_defs

from cpr_model import cpr_model
from util import extract_datasets, get_error_metrics,write_statistics_to_file,get_model_size,\
                 transform_dataset, transform_predictor, transform_response, inverse_transform_response

def generate_models(reg,cp_rank):
    model_list = []
    for i in reg:
        for j in cp_rank:
            model_list.append((i,j))
    return model_list

def main(args):
    timers = [0.]*3
    training_df = pd.read_csv('%s'%(args.training_file), index_col=0, sep=',')
    test_df = pd.read_csv('%s'%(args.test_file), index_col=0, sep=',')
    param_list = training_df.columns[[int(n) for n in args.input_columns.split(',')]].tolist()
    data_list = training_df.columns[[int(n) for n in args.data_columns.split(',')]].tolist()

    regularization_parameters = [float(n) for n in args.reg.split(',')]
    cp_ranks = [int(n) for n in args.cp_rank.split(',')]

    # Generate list of model types parameterized on hyper-parameters
    model_list = generate_models(regularization_parameters,cp_ranks)

    (training_configurations,training_data,training_set_size,\
            validation_configurations,validation_data,validation_set_size,\
            test_configurations,test_data,test_set_size,mode_range_min,mode_range_max)\
      = extract_datasets(training_df,test_df,param_list,data_list,args.training_set_size,\
          args.test_set_size,args.training_set_split_percentage,args.mode_range_min,args.mode_range_max)

    cell_spacing = [int(n) for n in args.cell_spacing.split(',')]
    assert(len(cell_spacing)==len(param_list))
    ngrid_pts = [int(n) for n in args.ngrid_pts.split(',')]
    assert(len(ngrid_pts)==len(param_list))
    custom_grid_pts = [int(n) for n in args.custom_grid_pts.split(',')]

    if (args.verbose):
        print("Location of training data: %s"%(args.training_file))
        print("Location of test data: %s"%(args.test_file))
        print("Location of output data: %s"%(args.output_file))
        print("args.input_columns: ",args.input_columns)
        print("args.data_columns: ",args.data_columns)
        print("param_list: ", param_list)
        print("reg: ",regularization_parameters)
        print("cp_rank: ",cp_ranks)
        print("model_list: ", model_list)
        print("cell_spacing: ", cell_spacing)
        print("ngrid_pts: ", ngrid_pts)

    start_time = time.time()
    opt_model_parameters = [-1,-1]
    opt_model = []
    opt_model_info = [np.inf]*4
    opt_error_metrics = [100000.]*16
    # Iterate over all model hyper-parameters (not including cell-count)
    for model_parameters in model_list:
        cp_rank = model_parameters[1]
        reg_lambda = model_parameters[0]
        start_time_solve = time.time()
        cpr_mod = cpr_model(ngrid_pts,[int(n) for n in args.interp_map.split(',')],cell_spacing,\
                    mode_range_min,mode_range_max,cp_rank,args.cp_rank_for_extrapolation,args.loss_function,\
                    reg_lambda,args.max_spline_degree,\
                    args.response_transform,custom_grid_pts,args.sweep_tol,args.max_num_sweeps,\
                    args.tol_newton,args.max_num_newton_iter,args.barrier_start,args.barrier_stop,\
                    args.barrier_reduction_factor,[int(n) for n in args.projection_set_size_threshold.split(',')] if len(args.projection_set_size_threshold)>0 else [],args.build_extrapolation_model)
        num_tensor_elements,density,loss1,loss2 = cpr_mod.fit(training_configurations,training_data)
        timers[0] += (time.time()-start_time_solve)
        model_predictions = []
        # Evaluate on validation set
        for k in range(validation_set_size):
            configuration = validation_configurations[k,:]*1.
            model_predictions.append(cpr_mod.predict(configuration))
        validation_error_metrics = get_error_metrics(validation_set_size,validation_configurations,validation_data,model_predictions)
        model_info = [num_tensor_elements,density,loss1,loss2]
        if (validation_set_size==0 or validation_error_metrics[2] < opt_error_metrics[2]):
            opt_model_parameters = copy.deepcopy(model_parameters)
            opt_model_info = copy.deepcopy(model_info)
            opt_model = copy.deepcopy(cpr_mod)
    timers[1] += (time.time()-start_time)

    model_size = get_model_size(opt_model,"CPR_Model.joblib")

    if (args.print_model_parameters):
        opt_model.print_parameters()

    start_time = time.time()
    model_predictions = []
    for k in range(test_set_size):
        configuration = test_configurations[k,:]*1.
        model_predictions.append(opt_model.predict(configuration))
    timers[2] += (time.time()-start_time)
    test_error_metrics = get_error_metrics(test_set_size,test_configurations,test_data,model_predictions,args.print_test_error)
    model_predictions = []
    for k in range(training_set_size):
        configuration = training_configurations[k,:]*1.
        model_predictions.append(opt_model.predict(configuration))
    training_error_metrics = get_error_metrics(training_set_size,training_configurations,training_data,model_predictions,0)
    write_statistics_to_file(args.output_file,test_error_metrics,training_error_metrics,timers,[training_set_size,validation_set_size,test_set_size],model_size,[len(ngrid_pts),'-'.join([str(i) for i in args.ngrid_pts.split(',')]),opt_model_parameters[0],opt_model_parameters[1],opt_model_info[0],opt_model_info[1],opt_model_info[2],opt_model_info[3],args.cp_rank_for_extrapolation,args.loss_function,args.sweep_tol,args.max_num_sweeps,args.tol_newton,args.max_num_newton_iter,args.barrier_start,args.barrier_stop,args.barrier_reduction_factor,'-'.join([str(i) for i in args.projection_set_size_threshold.split(',')])],["model:tensor_dim","model:ngrid_pts","model:reg","model:cprank","model:ntensor_elems","model:tensor_density","model:loss1","model:loss2","extrap_cp_rank","loss_function","sweep_tol","max_num_sweeps","tol_newton","max_num_newton_iter","barrier_start","barrier_stop","barrier_reduction_factor","projection_set_size_threshold"])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--write-header',
        type=int,
        default='0',
        metavar='int',
        help='Boolean decision whether to write column headers to CSV file (default: 0)')
    parser.add_argument(
        '--training-set-size',
        type=int,
        default=0,
        metavar='int',
        help='Size of training set (default: 0).')
    parser.add_argument(
        '--test-set-size',
        type=int,
        default=0,
        metavar='int',
        help='Size of test set (default: 0).')
    parser.add_argument(
        '--ngrid-pts',
        type=str,
        default='2',
        metavar='str',
        help='ID for discretization granularity of kernel configuration space (default: 2).')
    parser.add_argument(
        '--custom-grid-pts',
        type=str,
        default='0',
        metavar='str',
        help='Grid-point locations for any mode in order of modes with cell_spacing=2 (default: 0).')
    parser.add_argument(
        '--cell-spacing',
        type=str,
        default="1",
        metavar='str',
        help='ID for placement of grid-points constrained to a particular discretization granularity (not necessarily equivalent to sampling distribution of training dataset (default: 1).')
    parser.add_argument(
        '--training-set-split-percentage',
        type=float,
        default='0',
        metavar='float',
        help='Percentage of the training set used for model selection across hyper-parameter space (default: 0).')
    parser.add_argument(
        '--response-transform',
        type=int,
        default="1",
        metavar='int',
        help='Transformation to apply to runtime data (default: 1 (Log transformation)).')
    parser.add_argument(
        '--max-spline-degree',
        type=int,
        default="3",
        metavar='int',
        help='Maximum spline degree for extrapolation model (default: 1).')
    parser.add_argument(
        '--build-extrapolation-model',
        type=int,
        default="1",
        metavar='int',
        help='Signifies whether to build a separate model for extrapolation (default: 1).')
    parser.add_argument(
        '--reg',
        type=str,
        default='1e-4',
        metavar='str',
        help='regularization coefficient (default: 1e-5).')
    parser.add_argument(
        '--max-num-sweeps',
        type=int,
        default='100',
        metavar='str',
        help='Maximum number of sweeps of alternating minimization (default: 20).')
    parser.add_argument(
        '--sweep-tol',
        type=float,
        default='1e-5',
        metavar='float',
        help='Error tolerance for alternating minimization method (default: 1e-3).')
    parser.add_argument(
        '--barrier-start',
        type=float,
        default='1e1',
        metavar='float',
        help='Coefficient on barrier terms for initial sweep of Alternating Minimization via Newtons method (default: 100).')
    parser.add_argument(
        '--barrier-stop',
        type=float,
        default='1e-11',
        metavar='float',
        help='Coefficient on barrier terms for initial sweep of Alternating Minimization via Newtons method (default: 100).')
    parser.add_argument(
        '--barrier-reduction-factor',
        type=float,
        default='8',
        metavar='float',
        help='Divisor for coefficient on barrier terms for subsequent sweeps of Alternating Minimization via Newtons method (default: 1.25).')
    parser.add_argument(
        '--projection-set-size-threshold',
        type=str,
        default='',
        metavar='str',
        help='Comma-delimited list of minimum number of observations per hyperplane per tensor mode for which to construct model from (default: 8).')
    parser.add_argument(
        '--tol-newton',
        type=float,
        default='1e-3',
        metavar='float',
        help='Change (in factor matrix) tolerance within Newtons method (default: 1e-3).')
    parser.add_argument(
        '--max-num-newton-iter',
        type=int,
        default='40',
        metavar='float',
        help='Maximum number of iterations of Newtons method (default: 40)')
    parser.add_argument(
        '--cp-rank',
        type=str,
        default="3",
        metavar='str',
        help='Comma-delimited list of Canonical-Polyadic tensor decomposition ranks (default: 3).')
    parser.add_argument(
        '--cp-rank-for-extrapolation',
        type=int,
        default="1",
        metavar='int',
        help='Canonical-Polyadic tensor decomposition rank for use in extrapolation (default: 1).')
    parser.add_argument(
        '--loss-function',
        type=int,
        default="0",
        metavar='int',
        help='Loss function to optimize CPD Model for interpolation environment.')
    parser.add_argument(
        '--interp-map',
        type=str,
        default='',
        metavar='str',
        help='Comma-delimited list signifying which parameter ranges to interpolate (default: ).')
    parser.add_argument(
        '--training-file',
        type=str,
        default='',
        metavar='str',
        help='File path to training dataset.')
    parser.add_argument(
        '--test-file',
        type=str,
        default='',
        metavar='str',
        help='File path to test dataset.')
    parser.add_argument(
        '--output-file',
        type=str,
        default='',
        metavar='str',
        help='File path to write prediction results.')
    parser.add_argument(
        '--input-columns',
        type=str,
        default='',
        metavar='str',
        help='Comma-delimited list of column indices corresponding to each parameter witin a configuration for both training and test datasets.')
    parser.add_argument(
        '--data-columns',
        type=str,
        default='',
        metavar='str',
        help='Comma-delimited list of column indices corresponding to response (execution time) within training and test datasets (same format assumed for both).')
    parser.add_argument(
        '--mode-range-min',
        type=str,
        default='',
        metavar='str',
        help='Comma-delimited list of minimum values for each parameter within a configuration.')
    parser.add_argument(
        '--mode-range-max',
        type=str,
        default='',
        metavar='str',
        help='Comma-delimited list of maximum values for each parameter within a configuration.')
    parser.add_argument(
        '--print-model-parameters',
        type=int,
        default=0,
        metavar='int',
        help='Whether or not to print the factor matrix elements (default: 0).')
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true')
    parser.add_argument(
        '--print-test-error',
        type=int,
        default="0",
        metavar='int',
        help='')
    args, _ = parser.parse_known_args()
    main(args)
