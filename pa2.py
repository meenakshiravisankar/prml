import numpy as np
import functions as f
from scipy.stats import multivariate_normal

np.random.seed(seed=42)

# dataset 2 gives best accuracy with Model 5 in problem 1
data = np.loadtxt("../Datasets_PRML_A1/Dataset_2_Team_39.csv", delimiter=',', dtype=None)

# shuffling
np.random.shuffle(data)

# splitting into train, test - 80, 20 - and converting to numpy array
train_size = int(0.8*data.shape[0])
test_size = int(0.1*data.shape[0])

X_train = data[:train_size,:2]
y_train = data[:train_size,-1]

X_test = data[train_size:train_size+test_size,:2]
y_test = data[train_size:train_size+test_size,-1]

dataset_sizes = [100, 500, 1000, 2000, 4000]

lossfunction = np.array([[0,1,2],[1,0,1],[2,1,0]])

accuracies = []
for ds in dataset_sizes:
    # 20 replications
    test_accuracy = []
    for i in range(20):
        np.random.shuffle(X_train)
        np.random.shuffle(y_train)
        X = X_train[:ds]
        y = y_train[:ds]
        classes, prior = f.getPrior(y)
        means = np.array(f.getMLE(X, y))
        cov_rand = np.random.rand(1)*np.eye(2) # change this to actual covariance
        train_pred, train_acc =  f.getModel(X, y, means, cov_rand, lossfunction, prior, "bayes")
        test_pred, test_acc =  f.getModel(X_test, y_test, means, cov_rand, lossfunction, prior, "bayes")
        test_accuracy.append(test_acc)
    accuracies.append(f.mean(test_accuracy))

print(accuracies)
