import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix
import confusion_matrix as cf_mat
import matplotlib.pyplot as plt
import functions as f

# seeding
np.random.seed(seed=42)

# read dataset 3
data = np.loadtxt("../Datasets_PRML_A1/Dataset_3_Team_39.csv", delimiter=',',dtype=None)
lossfunction = np.array([[0, 1], [1, 0]])

# range of train dataset sizes
train_sizes = np.array([2,3,4,5,6,7,8,9,10,50,100,500,1000,3000])

np.savetxt("results/q3/sizes.txt",train_sizes,fmt="%.2f")

# shuffling
np.random.shuffle(data)
train_size = data.shape[0]
accuracy1 = []
for train_size in train_sizes :
    accuracy1.append(train_size)
    # considering 1, 2, or 3 features at a time
    for i in [1, 2, 3]:
        print()
        print("**********************\n")
        print("Number of features considered:", i)

        X_train = data[:train_size,:i]
        y_train = data[:train_size,-1]

        print("Size of train set = ", X_train.shape)

        # finding the prior class probabilities
        classes, prior = f.getPrior(y_train)

        # finding the class-wise means
        means = np.array(f.getMLE(X_train, y_train))

        # finding the class-wise covariances
        cov_rand = f.getCompleteCovMatrix(X_train, y_train)

        # obtaining the prediction and accuracy of our classifier
        prediction, accuracy =  f.getModel(X_train, y_train, means, cov_rand, lossfunction, prior, "bayes", "diff")
        print("Train accuracy {:.2f}".format(accuracy))
        accuracy1.append(1-accuracy/100)
accuracy1 = np.array(accuracy1).reshape(-1,4)
np.savetxt("results/q3/dataset3.txt",accuracy1,fmt="%.2f")



print()
print("**********************\n")

# read dataset 4
data = np.loadtxt("../Datasets_PRML_A1/Dataset_4_Team_39.csv", delimiter=',',dtype=None)

# shuffling
np.random.shuffle(data)
accuracy2 = []

for train_size in train_sizes :
    accuracy2.append(train_size)
    # considering 1, 2 or 3 features at a time
    for i in [1, 2, 3]:
        print()
        print("**********************\n")
        print("Number of features considered:", i)

        X_train = data[:train_size,:i]
        y_train = data[:train_size,-1]

        print("Size of train set = ", X_train.shape)

        # finding the prior class probabilities
        classes, prior = f.getPrior(y_train)

        # finding the class-wise means
        means = np.array(f.getMLE(X_train, y_train))

        # finding the class-wise covariances
        cov_rand = f.getCompleteCovMatrix(X_train, y_train)

        # obtaining the prediction and accuracy of our classifier
        prediction, accuracy =  f.getModel(X_train, y_train, means, cov_rand, lossfunction, prior, "bayes", "diff")
        print("Train accuracy {:.2f}".format(accuracy))
        accuracy2.append(1-accuracy/100)
accuracy2 = np.array(accuracy2).reshape(-1,4)
np.savetxt("results/q3/dataset4.txt",accuracy2,fmt="%.2f")

print()
print("**********************\n")
