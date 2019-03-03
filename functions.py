import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix
import confusion_matrix as cf_mat
import matplotlib.pyplot as plt
import math

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

def mvNormal(X, mu, sigma):
    det = np.linalg.det(sigma)
    n = len(mu)
    sigma_inv = np.linalg.inv(sigma)
    value = ((-0.5) * (X - mu) * sigma_inv * (X - mu)) / ((2 * math.pi)**(-n/2) * math.sqrt(det))
    return value

def mvNormalPDF(X, mu, sigma):
    vals = []
    for i in range(len(X)):
        vals.append(mvNormal(X[i], mu, sigma))
    return vals

def eval1DGaussian(X, mu, sigma):
    power = (np.sum((X-mu)*(X-mu)) * -1) / (2 * sigma * sigma)
    factor = 1/(math.sqrt(2 * math.pi) * sigma)
    value = factor * math.exp(power)
    return value

def getRisk(lossfunction, classConditional, prior) :
    return np.transpose(np.matmul(lossfunction,np.transpose(np.multiply(classConditional,prior))))

def getConditional(X, mu, sigma, mode):
    prob = []
    for class_val in range(mu.shape[0]) :
        if mode == "naive" :
            value = 1
            for feature in range(mu.shape[1]) :
                value *= eval1DGaussian(X[:,feature], mu[class_val][feature], sigma[feature][feature])
                # doubt: shouldn't we consider only those features that have been assigned this class_val??
        else :
            value = multivariate_normal.pdf(X, mean=mu[class_val], cov=sigma)
            #value =  mvNormalPDF(X, mu[class_val], sigma)
        prob.append(value)
    return np.transpose(np.array(prob))

def confusionMatrix(true_labels, pred_labels):
    n_classes = np.unique(true_labels).size
    s = (n_classes, n_classes)
    cm = np.zeros(s, dtype=int)
    for i in range(true_labels.size):
        true_index = int(true_labels[i])
        pred_index = int(pred_labels[i])
        cm[true_index][pred_index] += 1

    return cm

def getConfusion(y_test, prediction, name) :
    # confusion matrix for test
    # cnf_matrix = confusion_matrix(y_test, prediction)
    cnf_matrix = confusionMatrix(y_test, prediction)
    class_names = np.unique(prediction, return_counts=False)
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plt.figure()
    cf_mat.plot_confusion_matrix(cnf_matrix, classes=class_names,title=name)
    plt.savefig("results/"+name)
    # plt.show()
    return

# compute for test set
def getModel(X, y, means, cov, lossfunction, prior, mode) :
    classConditional = getConditional(X, means, cov, mode)
    risk = getRisk(lossfunction, classConditional, prior)
    prediction = np.argmin(risk, axis=1)
    accuracies = np.sum(prediction == y)/y.shape[0]*100
    return prediction, accuracies
