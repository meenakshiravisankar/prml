import numpy as np
import math
import random
import matplotlib.pyplot as plt

random.seed(42)

def getPolyfeatures(X,n) :
    return np.squeeze(np.transpose(np.array([X**i for i in range(n+1)])))

def getPolyfit(X,w) :
    return np.matmul(X,w)

def getWeights(X,y,ridge) :
    return np.matmul(np.matmul(np.linalg.inv(ridge*np.eye(X.shape[1]) + np.matmul(np.transpose(X),X)),np.transpose(X)),y)

def getEmpiricalRisk(y_train, y_pred, w, lamda):
    diff = y_train - y_pred
    ridge = lamda * sum(w*w)
    sqd = np.squeeze(np.array([x*x for x in diff]))
    return (sum(sqd) + ridge)

degrees = [1, 5, 9]
lambdas = [0.001, 0.01, 0.1]

for d in degrees:
    for myLambda in lambdas:
        empirical_risks = []
        for i in range(1000):
            samples = []
            X = np.random.uniform(low=-1,high=1,size=10).reshape(-1,1)
            y = np.exp(np.tanh(2*math.pi*X)) - X + np.random.normal(0,0.2)
            X_poly = getPolyfeatures(X, d)
            w = getWeights(X_poly, y, myLambda)
            y_pred = getPolyfit(X_poly, w)
            er = getEmpiricalRisk(y, y_pred, w, myLambda)
            empirical_risks.append(er)
        plt.hist(empirical_risks, bins=50,alpha=0.5)
        plt.show()
