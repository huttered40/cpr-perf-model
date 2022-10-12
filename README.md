# cpr-perf-model

Welcome!

This repository hosts a framework for high-dimensional performance modeling via CP tensor decomposition of Regular grids.

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
 All errors (besides `loss`) are with respect to the test set.
 
| Output  | Meaning |
| ------------- | ------------- |
| training_set_size | as specified in input |
| test_set_size | as specified in input |
| tensor_dim | as specified in input |
| ngrid_pts | as specified in input |
| cell_spacing | as specified in input |
| density | percentage of grid-cells tht have at least one sample |
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
 
 
Dependencies: CTF multilinear

Assumptions:
 - All configurations of parameters may be casted as non-negative floats.
 - Configurations of parameters may not contain categorical values like strings.
 - Training/Test data is comma-delimited.
