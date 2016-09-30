# LinearSVM

This package provides algorithms for solving the linear SVM problem.

## ADMM

Uses the [ADMM framework](https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf) to solve linear SVM in parallel.
The algorithm is developed by [Caoxie Zhang, Honglak Lee and Kang G. Shin](http://www.jmlr.org/proceedings/papers/v22/zhang12a/zhang12a.pdf).
For solving the subproblem the dual coordinate descent method is used.

Start a Julia session and create worker processes, e.g.

```
julia -p 4
```

```julia
using LinearSVM

# Load a dataset:
(train_x, train_y, test_x, test_y) = loadDataset("./spambase.data") # or whatever dataset you are using

# Distribute the dataset to the worker processes:
x,y = distributeData(train_x, train_y)

# run the ADMM algorithm:
w = admm(x,y,parallel=true)

# evaluate accuracy on test set:
testResult(w,test_x,test_y)
```

## Pegasos

The [Pegasos (primal estimated sub-gradient solver)](http://ttic.uchicago.edu/~nati/Publications/PegasosMPB.pdf) method by Shai Shalev-Shwartz, Yoram Singer, Nathan Srebro and Andrew Cotter.

```julia
using LinearSVM

(train_x, train_y, test_x, test_y) = loadDataset("./spambase.data") # or whatever dataset you are using
w = pegasos(train_x, train_y, lambda=0.1, batchsize=10, projection=true)
testResult(w,test_x,test_y)
```

## Dual Coordinate Descent

The [Dual Coordinate Descent](https://www.csie.ntu.edu.tw/~cjlin/papers/cddual.pdf) method by Cho-Jui Hsieh, Kai-Wei Chang, Chih-Jen Lin, S. Sathiya Keerthi and S. Sundararajan.


```julia
using LinearSVM

(train_x, train_y, test_x, test_y) = loadDataset("./spambase.data") # or whatever dataset you are using
w = dualcd(train_x, train_y,C=6.0, shrinking=true)
testResult(w,test_x,test_y)
```
