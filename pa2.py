import numpy as np
import functions as f
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# seeding
np.random.seed(seed=42)

# dataset 2 gives best accuracy with Model 5 in problem 1
data = np.loadtxt("../Datasets_PRML_A1/Dataset_2_Team_39.csv", delimiter=',', dtype=None)

# shuffling
np.random.shuffle(data)

# splitting into train, test - 80, 20 - and converting to numpy array
train_size = int(0.85*data.shape[0])
test_size = int(0.15*data.shape[0])
data_train = data[:train_size,:]
X_test = data[train_size:train_size+test_size,:2]
y_test = data[train_size:train_size+test_size,-1]

# X_train = data[:train_size,:2]
# y_train = data[:train_size,-1]
dataset_sizes = [x*100 for x in range(41)]
dataset_sizes[0] = 50

lossfunction = np.array([[0,1,2],[1,0,1],[2,1,0]])

yerr = np.zeros((2,len(dataset_sizes)))
idx = 0
accuracies = []
for ds in dataset_sizes:
    # 20 replications
    test_accuracy = []
    for i in range(20):
        np.random.shuffle(data_train)
        X = data_train[:ds,:2]
        y = data_train[:ds,-1]

        # training the classifier; obtaining train and test accuracies
        classes, prior = f.getPrior(y)
        means = np.array(f.getMLE(X, y))
        cov_rand = f.getCompleteCovMatrix(X, y)
        train_pred, train_acc =  f.getModel(X, y, means, cov_rand, lossfunction, prior, "bayes", "diff")
        test_pred, test_acc =  f.getModel(X_test, y_test, means, cov_rand, lossfunction, prior, "bayes", "diff")
        test_accuracy.append(test_acc)

    # getting the minimum and maximum for error bars
    yerr[0][idx] = np.abs(np.min(test_accuracy)-f.mean(test_accuracy))
    yerr[1][idx] = np.abs(np.max(test_accuracy)-f.mean(test_accuracy))
    idx+=1
    accuracies.append(f.mean(test_accuracy))

dataset_sizes = np.array(dataset_sizes)
yerr = np.array(yerr)
accuracies = np.array(accuracies)

# interpolating to find the number of training sample reqd to get 95.8% test accuracy
n_spl = interp1d(accuracies, dataset_sizes)
print(n_spl(95.8))
plt.plot(dataset_sizes, accuracies,'*')
plt.plot([0, n_spl(95.8)], [95.8, 95.8], 'g')
plt.plot([n_spl(95.8), n_spl(95.8)], [91, 95.8], 'g')
plt.errorbar(dataset_sizes, accuracies, yerr, ecolor='orange')
plt.xlabel("Dataset sizes")
plt.ylabel("Accuracies")
plt.margins(0)
plt.savefig("results/q2.png")
plt.show()
