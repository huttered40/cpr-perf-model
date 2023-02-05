kernel_type signifies the following:
  1. the domain from which the dataset is generated.
  2. how noisy each sample is

kernel types:
  0: [32,4096] with un-noisy data
  1: [128,524288] with un-noisy data
  10: [32,4096] with noisy data
  11: [128,524288] with noisy data

Change some of the pre-set variables in gemm.cpp / gemm_multinode.cpp
  to toggle the amount of noise per sample.
