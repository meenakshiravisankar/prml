# References
# 1) https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
#    To understand how to split the data into train, test and validation sets

##########################################################################

import csv
import random
import numpy as np
import math
from decimal import Decimal

def loadCsv(filename):
    rows = []
    with open(filename, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)

        # extracting field names through first row
        fields = next(csvreader)

        # extracting each data row one by one
        for row in csvreader:
            rows.append(row)

        new_rows = []
        for row in rows:
            temp = []
            for c in row:
                temp.append(float(c))
            new_rows.append(temp)
        return new_rows

def splitDataset(dataset):
    trainSize = int(len(dataset) * 0.7)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))

    testSize =  int(len(dataset) * 0.15)
    testSet = []
    copy1 = list(copy)
    while len(testSet) < testSize:
        index = random.randrange(len(copy1))
        testSet.append(copy1.pop(index))
    validationSet = copy1
    return [trainSet, testSet, validationSet]

def evalGaussian(X, mu, sigma):
    power = ((X - mu) * (X - mu) * (-1)) / (2 * sigma * sigma)
    factor = 1/(math.sqrt(2 * math.pi) * sigma)
    value = factor * math.exp(power)
    return value

def computeClassCase1(L, means, variances, X):
    for alpha_i in range(0, 3):
        mean = means[alpha_i]
        var = variances[alpha_i]
        sd = [math.sqrt(v) for v in var]
        f0 = evalGaussian(X[0], mean[0], sd[0])
        f1 = evalGaussian(X[1], mean[1], sd[1])
        sub = np.subtract(X, mean)
        temp = [x*x for x in sub]
        sum = np.sum(temp)
        disc = (-0.5 * sum) + float(Decimal(f0 * f1).ln())
        if disc > 0:
            return alpha_i
    return random.randrange(3)

filename = "../Datasets_PRML_A1/Dataset_1_Team_39.csv"
dataset = loadCsv(filename)
train, test, val = splitDataset(dataset)
print("The number of training samples is %d, test samples, is %d and validation samples is %d" %(len(train), len(test), len(val)))

n_rows = len(dataset)
n_cols = len(dataset[0])

# number of features
n_feat = n_cols - 1

# number of class labels
m = 0
m_max = 0

if n_cols > 1:
    for r in dataset:
        if r[n_cols-1] > m:
            m_max = r[n_cols-1]

m = int(m_max + 1)
classlabels = [0] * m

print("Number of rows = %d" %(n_rows))
print("Number of columns = %d" %(n_cols))
print("Number of features = %d" %(n_feat))
print("Number of class labels = %d (0 to %d)" %(m, m_max))

# finding prior probabilities
for row in dataset:
    classlabels[int(row[n_cols-1])] = classlabels[int(row[n_cols-1])] + 1

priors = [x / n_rows for x in classlabels]

print(priors)

# finding class conditional density parameters by MLE
# MLE for mean is sample mean
ccd_tot = np.zeros((m,1), dtype = float, order = 'C')
ccd_means = np.zeros((m, n_feat), dtype = float, order = 'C')
for row in dataset:
    cl = int(row[n_cols-1])
    ccd_means[cl] = np.add(ccd_means[cl], list(row[0:n_cols-1]))
    ccd_tot[cl] = ccd_tot[cl] + 1

for i in range(0, m):
    for j in range (0, n_feat):
     ccd_means[i][j] = ccd_means[i][j] / ccd_tot[i]
print(ccd_means)

# MLE for variance is (1/n)sigma(Xi - mean)^2
ccd_variances = np.zeros((m, n_feat), dtype = float, order = 'C')
for row in dataset:
    cl = int(row[n_cols-1])
    temp = np.subtract(list(row[0:n_cols-1]), ccd_means[cl])
    ccd_variances[cl] = [x*x for x in temp]

for i in range(0, m):
    for j in range (0, n_feat):
     ccd_variances[i][j] = ccd_variances[i][j] / ccd_tot[i]
print(ccd_variances)

# for dataset 1, covariance matrix is given to be I
covariances = np.eye(n_feat)
print(covariances)

# given loss function
loss_fn = np.matrix([[0,1,2],[1,0,1],[2,1,0]])
print(loss_fn)

total = 0
for row in dataset:
    actual_class = int(row[n_cols-1])
    X = [int(x) for x in row[0:n_cols-1]]
    predicted_class = computeClassCase1(loss_fn, ccd_means, ccd_variances, X)
    if actual_class == predicted_class:
        total = total + 1

accuracy = (total / n_rows) * 100
print("Accuracy of predictions = %f" %(accuracy))
