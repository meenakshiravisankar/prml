import numpy as np
import math
import random
import matplotlib.pyplot as plt
import scipy.stats as stats

# seedings
random.seed(42)

# returns the polynomial terms, raised to each degree from 0 to n
def getPolyfeatures(X,n) :
    return np.squeeze(np.transpose(np.array([X**i for i in range(n+1)])))

# returns the fit with the given coefficients
def getPolyfit(X,w) :
    return np.matmul(X,w)

# returns the weights corresponding to solution of the famous equation
def getWeights(X,y,ridge) :
    return np.matmul(np.matmul(np.linalg.inv(ridge*np.eye(X.shape[1]) + np.matmul(np.transpose(X),X)),np.transpose(X)),y)

# calculates the empirical risk (average of the cost function)
def getEmpiricalRisk(y_train, y_pred, w, lamda):
    n = y_train.shape[0]
    diff = y_train - y_pred
    ridge = lamda * sum(w*w)
    sqd = np.squeeze(np.array([x*x for x in diff]))
    error = sum(sqd) + ridge
    emp_risk = (error+0.0)/n
    return emp_risk

# range of degrees of the polynomial
degrees = [1, 5, 9]

# range of the regularization coefficients
lambdas = [0.001, 0.01, 0.1]

fig = 331
for d in degrees:
    for myLambda in lambdas:
        empirical_risks = []
        plt.subplot(fig)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.title("degree = " + str(d) + ", lambda = " + str(myLambda),fontsize=8)
        fig+=1
        # repeating for 1000 iterations
        for i in range(1000):
            # sampling 10 data points
            X = np.random.uniform(low=-1,high=1,size=10).reshape(-1,1)

            # adding the noise factor
            noise = np.random.normal(0,np.sqrt(0.2),10).reshape(-1,1)

            # obtaining the corresponding y values
            func = np.exp(np.tanh(2*math.pi*X)) - X
            y = func + noise

            data = np.append(X,y,axis=1)
            np.random.shuffle(data)
            X = data[:,0]
            y = data[:,1]

            # performing ridge regression and getting the empirical risk
            X_poly = getPolyfeatures(X, d)
            w = getWeights(X_poly, y, myLambda)
            y_pred = getPolyfit(X_poly, w)
            er = getEmpiricalRisk(y, y_pred, w, myLambda)
            empirical_risks.append(er)

        # plotting the histogram outline and the smooth curve corresponding to it
        density = stats.gaussian_kde(empirical_risks)
        n, curve, _ = plt.hist(empirical_risks, bins=100, alpha=0.5, histtype=u'step', density=True)
        plt.plot(curve, density(curve))
plt.suptitle("Empirical risks histogram",fontsize=8)
plt.savefig("results/degree_vs_lambda")
