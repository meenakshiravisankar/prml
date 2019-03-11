import numpy as np
import functions as f
from scipy.stats import multivariate_normal

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

def getConditionalSameCov(X, mu, sigma, mode):
    prob = []
    for class_val in range(mu.shape[0]) :
        if mode == "naive" :
            value = 1
            for feature in range(mu.shape[1]) :
                value *= (multivariate_normal.pdf(X[:,feature],mean=mu[class_val][feature],cov=sigma[feature][feature]))
        else :
            value =  multivariate_normal.pdf(X,mean=mu[class_val],cov=sigma)
        prob.append(value)
    return np.transpose(np.array(prob))

def getConditionalDiffCov(X, mu, sigma, mode):
    prob = []
    for class_val in range(mu.shape[0]) :
        if mode == "naive" :
            value = 1
            for feature in range(mu.shape[1]) :
                value *= (multivariate_normal.pdf(X[:,feature],mean=mu[class_val][feature],cov=sigma[class_val][feature][feature]))
        else :
            value =  multivariate_normal.pdf(X,mean=mu[class_val],cov=sigma[class_val])
        prob.append(value)
    return np.transpose(np.array(prob))

def getRisk(lossfunction, classConditional, prior) :
    return np.transpose(np.matmul(lossfunction,np.transpose(np.multiply(classConditional,prior))))

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

# dataset 2 gives best accuracy with Model 5 in problem 1
data = np.loadtxt("../Datasets_PRML_A1/Dataset_2_Team_39.csv", delimiter=',', dtype=None)

# shuffling
np.random.shuffle(data)

# splitting into train, test - 80, 20 - and converting to numpy array
train_size = int(0.8*data.shape[0])
test_size = int(0.2*data.shape[0])

X_train = data[:train_size,:2]
y_train = data[:train_size,-1]

X_test = data[train_size:train_size+test_size,:2]
y_test = data[train_size:train_size+test_size,-1]

dataset_sizes = [100, 500, 1000, 2000, 4000]
#dataset_sizes = np.linspace(100,4000,39).astype(int)

lossfunction = np.array([[0,1,2],[1,0,1],[2,1,0]])

accuracies = []
for ds in dataset_sizes:
    # 20 replications
    test_accuracy = []
    for i in range(20):
        X = X_train[:ds]
        y = y_train[:ds]
        classes, prior = getPrior(y)
        means = np.array(getMLE(X, y))
        cov_rand = getCompleteCovMatrix(X, y)
        train_pred, train_acc =  getModel(X, y, means, cov_rand, lossfunction, prior, "bayes", "diff")
        test_pred, test_acc =  getModel(X_test, y_test, means, cov_rand, lossfunction, prior, "bayes", "diff")
        test_accuracy.append(test_acc)
    accuracies.append(mean(test_accuracy))

print(accuracies)
