# cpr-perf-model

Welcome!

This repository hosts a Python framework for high-dimensional performance modeling via CP tensor decomposition.
Modeling domains are characterized as multi-dimensional regular grid.
Performance data sampled across these grid comprise multi-dimensional arrays (i.e., tensors).

This framework configures canonical-polyadic (CP) tensor decomposition models from provided performance data and
leverages high-performance tensor computation software publically available within the Cyclops Tensor Framework.

Performance data is specified separately as training data within file `training_file` and test data within file `test_file`.
Parameters and data within these files must be comma-delimited and castable to non-negative floats (e.g., categorical parameters must first be mapped onto a non-negative real number scale).
An example is provided below, in which `input_columns`=0,1,2 and `data_columns`=3.
```
m,n,k,runtime
0,89,61,4075,0.00237545
1,1075,34,2247,0.00800553
2,1344,109,845,0.00987118
3,1968,216,1765,0.0497626
4,293,64,187,0.00029943
5,288,716,425,0.00615035
...
```
`training_set_size` samples are randomly selected from `training_file` and used to optimize CP decompositions.
Similarly, `test_set_size` samples are randomly selected from `test_file` and used to evaluate optimized models.
However, a random subset of the randomly selected test samples of size 100\% x `test_set_split_percentage` x `test_set_size` will partition the data within `test_file` to reserve data for hyper-parameter selection.
The remaining partition will be used to evaluate the configured CP decomposition model.

Minimum and maximum values can be specified as comma-delimited lists for each benchmark parameter within `mode_range_min` and `mode_range_max`, respectively.
If left unspecified, the range of each parameter will be deduced from the training data.

A number of model parameters govern CP decomposition performance model optimization, including `cell_spacing`, `ngrid_pts`, `response_transform`, `interp_map`, `nals_sweeps`, `reg`, and `cp_rank`.
`cell_spacing`, `ngrid_pts`, and `interp_map` each take a comma-delimited list, the size of which equates to the number of benchmark parameters.
`cell_spacing`=0,1 signifies that along the first and second benchark parameters, uniform spacing and geometric spacing is used to partition the ranges of the corresponding parameters, respectively.
`ngrid_pts` then specifies the number of grid-points to place along the range of each parameter (including boundaries).
`interp_map` specifies which tensor modes (equivalently dimensions of the underlying regular grid) to interpolate during inference time using the configured CP decomposition model.
Users may set `response_transform`=0 to use raw execution data and `response_transform`=1 to apply a logarithmic transformation.
`cp_rank` specifies the CP rank of the model, `reg` specifies the regularization parameter in the underlying objective function, and `nals_sweeps` specifies the maximum number of sweeps of the alternating least-squares algorithm used to optimize the CP decomposition.

A complete list of runtime arguments is provided below:

| Argument  | Meaning |
| ------------- | ------------- |
| training_file | Full path to csv file that stores training set |
| test_file | Full path to csv file that stores test set |
| output_file | Full path to csv file to write results |
| input_columns | Comma-delimited list of column indices corresponding to benchmark parameters |
| data_columns | Column index corresponding to execution times |
| training_set_size | Number of samples to use from specified training set |
| test_set_size | Number of samples to use from specified test set |
| test_set_split_percentage | Percentage of test-set samples to use for hyper-parameter selection |
| mode_range_min | Comma-delimited list of minimum values taken by each benchmark parameter  |
| mode_range_max | Comma-delimited list of maximum values taken by each benchmark parameter |
| cell_spacing | Comma-delimited list specifying the spacing between grid-points (0: Uniform spacing, 1: Geometric spacing) |
| ngrid_pts | Comma-delimited list specifying the number of grid-points (including end-points) along each dimension |
| response_transform | Whether or not to transform execution data (0: No transformation to execution data, 1: Logarithm transformation to execution data) |
| interp_map | Comma-delimited list specifying which tensor modes (equivalently grid dimensions) about which to interpolate (0: No interpolation, 1: Interpolate) |
| nals_sweeps | Number of sweeps of the Alternating Least Squares algorithm |
| reg | Regularization parameter |
| cp_rank | Canonical-Polyadic tensor decomposition rank |
 
 Example for a 3-parameter kernel and 3-dimensional tensor:
 ```
  ibrun -n 1 python cpr.py --test_set_size 1000 --test_set_split_percentage 0.1 --interp_map 1,1,1 --nals_sweeps 100 --reg 1e-5 --response_transform 1 --cp_rank 3 --training_set_size 65536 --training_file 'gemm-train.csv' --test_file 'gemm-test.csv' --output_file 'cpr-results.csv' --input_columns 0,1,2 --data_columns 3 --mode_range_min 32,32,32 --mode_range_max 4096,4096,4096 --cell_spacing 1,1,1 --cell_counts 4,4,4
 ```
 
 Output data containing loss on training data, error metrics on test data, and model configuration execution times is written to `output_file`.
 All errors (besides `loss`) are with respect to the test set.
 
 A complete list of the output data is provided below:
 
| Output  | Meaning |
| ------------- | ------------- |
| training_set_size | as specified in input |
| test_set_size | as specified in input |
| tensor_dim | as specified in input |
| ngrid_pts | as specified in input |
| cell_spacing | as specified in input |
| density | percentage of grid-cells that have at least one sample |
| response_transform | as specified in input |
| reg | as specified in input |
| nals_sweeps | as specified in input |
| cp_rank | as specified in input |
| interp_map | as specified in input |
| loss | mean squared error on training data |
| mlogq | arithmetic mean of log-absolute accuracy ratios |
| mlogq2 | arithmetic mean of log-squared accuracy ratios |
| gmre | geometric mean of relative error |
| mape | mean absolute percentage error |
| smape |symmetric mean absolute percentage error |
| tensor_gen_time | time to generate tensor from data |
| model_config_time | time to configure model |
