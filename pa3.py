import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix
import confusion_matrix as cf_mat
import matplotlib.pyplot as plt

np.random.seed(seed=42)

def getPrior(y) :
    unique, counts = np.unique(y, return_counts=True)
    return unique, np.array(counts/sum(counts)).reshape(1,-1)

# According to MLE, estimate mean of distribution using sample mean
def getMLE(X,y) :
    unique = np.unique(y, return_counts=False)
    means = []
    for class_val in unique :
        means.append(np.mean(X[np.where(y==class_val)],axis=0))
    return means


def getRisk(lossfunction, classConditional, prior) :
    return np.transpose(np.matmul(lossfunction,np.transpose(np.multiply(classConditional,prior))))

def getConditionalSameCov(X, mu, sigma, mode):
    prob = []
    for class_val in range(mu.shape[0]) :
        if mode == "naive" :
            value = 1
            for feature in range(mu.shape[1]) :
                value *= (multivariate_normal.pdf(X[:,feature],mean=mu[class_val][feature],cov=sigma[feature][feature], allow_singular=True))
        else :
            value =  multivariate_normal.pdf(X,mean=mu[class_val],cov=sigma, allow_singular=True)
        prob.append(value)
    return np.transpose(np.array(prob))

def getConditionalDiffCov(X, mu, sigma, mode):
    prob = []
    for class_val in range(mu.shape[0]) :
        if mode == "naive" :
            value = 1
            for feature in range(mu.shape[1]) :
                value *= (multivariate_normal.pdf(X[:,feature],mean=mu[class_val][feature],cov=sigma[class_val][feature][feature], allow_singular=True))
        else :
            value =  multivariate_normal.pdf(X,mean=mu[class_val],cov=sigma[class_val], allow_singular=True)
        prob.append(value)
    return np.transpose(np.array(prob))

# compute for test set
def getModel(X, y, means, cov, lossfunction, prior, mode, covmode) :
    if covmode == "same":
        classConditional = getConditionalSameCov(X, means, cov, mode)
    else:
        classConditional = getConditionalDiffCov(X, means, cov, mode)
    risk = getRisk(lossfunction, classConditional, prior)
    prediction = np.argmin(risk, axis=1)
    accuracies = np.sum(prediction == y)/y.shape[0]*100
    return prediction, accuracies

def mean(X):
    n = len(X)
    mean_x = 0.0
    for i in range(n):
        mean_x += X[i]
    mean_x = mean_x/n
    return mean_x

def getCovariance(X1, X2):
    Z = []
    n = len(X1)
    for i in range(n):
        Z.append(X1[i]*X2[i])
    e_x1x2 = mean(Z)
    e_x1 = mean(X1)
    e_x2 = mean(X2)
    cov = e_x1x2 - (e_x1*e_x2)
    return cov

def getCovMatrix(X):
    n = len(X)
    cov_mat = []
    for i in range(n):
        for j in range(n):
            cov_mat.append(getCovariance(X[i], X[j]))
    return np.reshape(cov_mat, (n, n))

def getCompleteCovMatrix(X, y):
    unique = np.unique(y, return_counts=False)
    covs = []
    for class_val in unique :
        covs.append(getCovMatrix(np.transpose(X[np.where(y==class_val)])))
    return covs

# read dataset 3
data = np.loadtxt("../Datasets_PRML_A1/Dataset_3_Team_39.csv", delimiter=',', dtype=None)
lossfunction = np.array([[0, 1], [1, 0]])

# # train, test, validation set sizes - 70, 15, 15
# train_size = int(0.7*data.shape[0])
# val_size = int(0.15*data.shape[0])
# test_size = int(0.15*data.shape[0])

# shuffling
np.random.shuffle(data)
# we thought that changing the train size may have an impact on the training error with
# varying number of features, however, I tried with range of train_size from 10 to 3000
# and each time, the accuracy is only increasing...
train_size = data.shape[0]

for i in [1, 2, 3]:
    print()
    print("**********************\n")
    print("Number of features considered:", i)

    X_train = data[:train_size,:i]
    y_train = data[:train_size,-1]

    # X_val = data[train_size:train_size+val_size,:i]
    # y_val = data[train_size:train_size+val_size,-1]
    #
    # X_test = data[train_size+val_size:train_size+val_size+test_size,:i]
    # y_test = data[train_size+val_size:train_size+val_size+test_size,-1]
    #
    # print("Size of train, validation and test sets",X_train.shape,X_val.shape,X_test.shape)

    print("Size of train set = ", X_train.shape)
    classes, prior = getPrior(y_train)
    means = np.array(getMLE(X_train, y_train))

    # accuracies = []

    cov_rand = getCompleteCovMatrix(X_train, y_train)
    prediction, accuracy =  getModel(X_train, y_train, means, cov_rand, lossfunction, prior, "bayes", "diff")
    print("Train accuracy {:.2f}".format(accuracy))
    # accuracies.append(accuracy)
    # prediction, accuracy =  getModel(X_val, y_val, means, cov_rand, lossfunction, prior, "bayes", "diff")
    # print("Validation accuracy {:.2f}".format(accuracy))
    # prediction, accuracy =  getModel(X_test, y_test, means, cov_rand, lossfunction, prior, "bayes", "diff")
    # print("Test accuracy {:.2f}".format(accuracy))

print()
print("**********************\n")

# read dataset 4
data = np.loadtxt("../Datasets_PRML_A1/Dataset_4_Team_39.csv", delimiter=',', dtype=None)

# shuffling
np.random.shuffle(data)

for i in [1, 2, 3]:
    print()
    print("**********************\n")
    print("Number of features considered:", i)

    X_train = data[:train_size,:i]
    y_train = data[:train_size,-1]

    # X_val = data[train_size:train_size+val_size,:i]
    # y_val = data[train_size:train_size+val_size,-1]
    #
    # X_test = data[train_size+val_size:train_size+val_size+test_size,:i]
    # y_test = data[train_size+val_size:train_size+val_size+test_size,-1]
    #
    # print("Size of train, validation and test sets",X_train.shape,X_val.shape,X_test.shape)

    print("Size of train set = ", X_train.shape)
    classes, prior = getPrior(y_train)
    means = np.array(getMLE(X_train, y_train))

    # accuracies = []

    cov_rand = getCompleteCovMatrix(X_train, y_train)
    prediction, accuracy =  getModel(X_train, y_train, means, cov_rand, lossfunction, prior, "bayes", "diff")
    print("Train accuracy {:.2f}".format(accuracy))
    # accuracies.append(accuracy)
    # prediction, accuracy =  getModel(X_val, y_val, means, cov_rand, lossfunction, prior, "bayes", "diff")
    # print("Validation accuracy {:.2f}".format(accuracy))
    # prediction, accuracy =  getModel(X_test, y_test, means, cov_rand, lossfunction, prior, "bayes", "diff")
    # print("Test accuracy {:.2f}".format(accuracy))

print()
print("**********************\n")
