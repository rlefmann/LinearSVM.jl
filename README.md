# LinearSVM

This package provides algorithms for solving the linear SVM problem.

## ADMM

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

## Dual Coordinate Descent
