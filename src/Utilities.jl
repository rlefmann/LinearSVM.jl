"""
   `loadDataset(filepath)`
    
Reads a dataset from a csv file located at `filepath`.
Each row of the dataset should be an
example, each column a feature. The last column are the targets, which
are assumed to be binary labels 0 and 1.
A column of ones is added to the beginning of the dataset and the set is
equally split into training and test set.

### Returns
 * train_x: the data for training
 * train_y: the labels of the training data
 * test_x: the data for testing
 * test_y: the labels of the test data
"""
function loadDataset(filepath)
    ## load dataset
    data = readdlm(filepath, ',')

    ## preprocess dataset
    print("preprocessing dataset...\t")
    tic()
    # split into features and labels:
    x = data[:,1:end-1]
    y = data[:,end]
    # replace 0 entries with -1
    y[y .== 0] = -1
    # add a column of ones to the beginning of x
    # because we encorporated b into w
    x0 = ones(size(x,1))
    x = hcat(x0,x)

    # seed for random number generator:
    srand(1234)
    # shuffle the training data:
    perm = randperm(size(x,1))
    x = x[perm,:]
    y = y[perm]

    # use every second datapoint for training
    # TODO: use more data for training
    train_x = x[1:2:end,:]
    train_y = y[1:2:end]
    test_x = x[2:2:end,:]
    test_y = y[2:2:end]

    @printf "finished after %.3f seconds\n" toq()

    return (train_x,train_y,test_x,test_y)
end


"""
   `testResult(w, test_x, test_y)`

Measures the classification performance of a hyperplane with normal
vector `w` on test data.

### Returns
 * `acc`: accuracy on the test data set
"""
function testResult(w, test_x, test_y)
    predicted_y = sign(test_x*w)
    acc = mean((predicted_y .== test_y))
    @printf "Accuracy: %.2f%%\n" acc*100
    return acc
end
