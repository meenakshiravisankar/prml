# Reference for plot: https://pythonspot.com/matplotlib-scatterplot/
import numpy as np
import functions as f
import math
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

random.seed(42)
colors = (0,0,0)
area = np.pi*3

def getPolyfeatures(X,n) :
    return np.squeeze(np.transpose(np.array([X**i for i in range(n+1)])))

def getPolyfit(X,w) :
    return np.matmul(X,w)

def getAllPolyfit(X,w) :
    polys = []
    for i in range(X.shape[0]):
        polys.append(getPolyfit(X[i], w))
    return np.array(polys)

def getWeights(X,y,ridge) :
    return np.matmul(np.matmul(np.linalg.inv(ridge*np.eye(X.shape[1]) + np.matmul(np.transpose(X),X)),np.transpose(X)),y)

def getAllWeights(X,y,ridge) :
    weights = []
    for i in range(X.shape[0]):
        weights.append(getWeights(X[i], y, ridge))
    return np.array(weights)

def getEmpiricalRisk(y_train, y_pred):
    diff = y_train - y_pred
    sqd = np.squeeze(np.array([x*x for x in diff]))
    return sum(sqd)

def kmeans(dataset, k):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(dataset)
    return kmeans.cluster_centers_

# Loading dataset
data = np.loadtxt("../Datasets_PRML_A1/train100.txt", delimiter=' ', dtype=None)
np.random.shuffle(data)
# Settings
degrees = [1,3,6,9]
ridges = [10**(-13+x) for x in range(14)]
ridges[0] = 0
sizes = [10,20,30,40,50,60,70]
variances = []
num_gauss = [5]

for k in num_gauss:
    means = np.transpose(kmeans(data, k))[-1]
    print(means)

    X_train = data[:,:2]
    y_train = data[:,-1]
    w = []
    X_poly = getPolyfeatures(X_train, 3)
    w = getAllWeights(X_poly, y_train, 0)
    y_pred = getAllPolyfit(X_poly, np.transpose(w))
    # compute rms error

    # controlling the overfitting by using Ridge Regression
    lambdas = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    for lb in lambdas:
        w = getAllWeights(X_poly, y_train, lb)
        y_pred = getAllPolyfit(X_poly, np.transpose(w))
        # compute rms error

    # controlling the overfitting by increasing the training data size
    data = np.loadtxt("../Datasets_PRML_A1/train1000.txt", delimiter=' ', dtype=None)

    X_train = data[:,:2]
    y_train = data[:,-1]
    w = []
    X_poly = getPolyfeatures(X_train, 3)
    w = getAllWeights(X_poly, y_train, 0)
    y_pred = getAllPolyfit(X_poly, np.transpose(w))
    # compute rms error
