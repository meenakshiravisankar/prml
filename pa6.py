import numpy as np
import math
import random
import matplotlib.pyplot as plt
import scipy.stats as stats

random.seed(42)

def getPolyfeatures(X,n) :
    return np.squeeze(np.transpose(np.array([X**i for i in range(n+1)])))

def getPolyfit(X,w) :
    return np.matmul(X,w)

def getWeights(X,y,ridge) :
    return np.matmul(np.matmul(np.linalg.inv(ridge*np.eye(X.shape[1]) + np.matmul(np.transpose(X),X)),np.transpose(X)),y)

def getEmpiricalRisk(y_train, y_pred, w, lamda):
    n = y_train.shape[0]
    diff = y_train - y_pred
    ridge = lamda * sum(w*w)
    sqd = np.squeeze(np.array([x*x for x in diff]))
    error = sum(sqd) + ridge
    emp_risk = (error+0.0)/n
    return emp_risk

degrees = [1, 5, 9]
lambdas = [0.001, 0.01, 0.1]
fig = 331
for d in degrees:
    for myLambda in lambdas:
        empirical_risks = []
        plt.subplot(fig)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.title("degree = " + str(d) + ", lambda = " + str(myLambda),fontsize=8)
        fig+=1
        for i in range(1000):
            X = np.random.uniform(low=-1,high=1,size=10).reshape(-1,1)
            noise = np.random.normal(0,np.sqrt(0.2),10).reshape(-1,1)
            func = np.exp(np.tanh(2*math.pi*X)) - X
            y = func + noise
            data = np.append(X,y,axis=1)
            np.random.shuffle(data)
            X = data[:,0]
            y = data[:,1]
            X_poly = getPolyfeatures(X, d)
            w = getWeights(X_poly, y, myLambda)
            y_pred = getPolyfit(X_poly, w)
            er = getEmpiricalRisk(y, y_pred, w, myLambda)
            empirical_risks.append(er)
        density = stats.gaussian_kde(empirical_risks)
        n, curve, _ = plt.hist(empirical_risks, bins=100, alpha=0.5, histtype=u'step', density=True)
        plt.plot(curve, density(curve))
plt.suptitle("Empirical risks histogram",fontsize=8)
plt.savefig("results/degree_vs_lambda")
