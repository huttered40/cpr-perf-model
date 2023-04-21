# Reproducibility Initiative

We provide scripts to evaluate ten performance modeling methods and to reproduce each figure provided in [https://arxiv.org/abs/2210.10184].

## Methods
The modeling methods (and corresponding hyper-parameters) we evaluate include the following:
1. Multivariate Adaptive Regression Splines [https://contrib.scikit-learn.org/py-earth/content.html#multivariate-adaptive-regression-splines]
    - maximum spline degree
2. Sparse Grid Regression [https://sgpp.sparsegrids.org]
    - sparse grid level
    - number of local grid refinements
    - number of adaptive grid-points added per local refinement
3. Multilayer perceptron [https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html]
    - number of layers
    - layer size
    - activation function
4. Gradient Boosting [https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html]
    - number of regression trees
    - maximum depth per regression tree
5. Random Forest Regression [https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html]
    - number of regression trees
    - maximum depth per regression tree
6. Extremely Randomized Tree Regression [https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html]
    - number of regression trees
    - maximum depth per regression tree
7. k-Nearest Neighbors [https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html]
    - number of neighbors
8. Support Vector Machines [https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html]
    - kernel
9. Gaussian Process Regression [https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html]
    - kernel
10. Canonical-Polyadic Decomposition of Regular Grids (our proposed method)
    - CP rank
    - number of grid-points placed along the ranges of each benchmark parameter

For specifics regarding the specific values per hyper-parameter that we evaluate, since the linked paper [https://arxiv.org/abs/2210.10184].

The external libraries we use to evaluate the methods listed above include:
1. PyEarch (version 0.1.0)
    - Multivariate Adaptive Regression Splines
3. SG++ (version 3.3.1)
    - Sparse Grid Regression
5. Scikit-learn (version 1.0.2)
    - Multilayer perceptron
    - Gradient Boosting
    - Random Forest Regression
    - Extremely Randomized Tree Regression
    - k-Nearest Neighbors
    - Support Vector Machines
    - Gaussian Process Regression

We provide Python programs to quickly evaluate each method (assuming the aforementioned dependencies are installed locally) here: [https://github.com/huttered40/cpr-perf-model/tree/main/src_python/alternative_models]
These programs are invoked within these scripts.

We provide the datasets we utilize here: [https://github.com/huttered40/app_ed/tree/main/datasets/stampede2].
Users must change the scripts to specify the correct path to each dataset and program.
See the corresponding README [https://github.com/huttered40/app_ed] for descriptions of each application or kernel benchmark parameter.

## Navigation of these scripts
The following loop structure explains this file directory:

```
for fig in [3,4,5,6,7,8]:
    for benchmark in [geqrf,gemm,bcast,exafmm,amg,kripke]:
        for method in [mars,sgr,mlp,gb,rf,et,knn,svm,gp,cpr]:
            Execute {method}.sh in directory cpr-perf-model/reproducibility/figure_{fig}/{benchmark}/
```
