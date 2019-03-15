import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix
import confusion_matrix as cf_mat
import matplotlib.pyplot as plt

# returns prior class probabilities
def getPrior(y) :
    unique, counts = np.unique(y, return_counts=True)
    return unique, np.array(counts/sum(counts)).reshape(1,-1)

# according to MLE, estimates mean of distribution using sample mean
def getMLE(X,y) :
    unique = np.unique(y, return_counts=False)
    means = []
    for class_val in unique :
        means.append(np.mean(X[np.where(y==class_val)],axis=0))
    return means

# calculates the posterior (numerator only) using loss function, class conditional density and prior
def getRisk(lossfunction, classConditional, prior) :
    return np.transpose(np.matmul(lossfunction,np.transpose(np.multiply(classConditional,prior))))

# calculates class conditional density for the case of same covariance for all classes
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

# claculates class conditional density for the case of different covariance for each class
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

# computes prediction and accuracies of classifier
def getModel(X, y, means, cov, lossfunction, prior, mode, covmode) :
    if covmode == "same":
        classConditional = getConditionalSameCov(X, means, cov, mode)
    else:
        classConditional = getConditionalDiffCov(X, means, cov, mode)
    risk = getRisk(lossfunction, classConditional, prior)
    prediction = np.argmin(risk, axis=1)
    accuracies = np.sum(prediction == y)/y.shape[0]*100
    return prediction, accuracies

# finds mean of a feature vector
def mean(X):
    n = len(X)
    mean_x = 0.0
    for i in range(n):
        mean_x += X[i]
    mean_x = mean_x/n
    return mean_x

# finds covariance of two feature vectors
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

# gets covariance matrix for a dataset X, considering equal covariance for all classes
def getCovMatrix(X):
    n = len(X)
    cov_mat = []
    for i in range(n):
        for j in range(n):
            cov_mat.append(getCovariance(X[i], X[j]))
    return np.reshape(cov_mat, (n, n))

# gets covariance matrix for a dataset X,y considering different covariance for each class
def getCompleteCovMatrix(X, y):
    unique = np.unique(y, return_counts=False)
    covs = []
    for class_val in unique :
        covs.append(getCovMatrix(np.transpose(X[np.where(y==class_val)])))
    return covs
