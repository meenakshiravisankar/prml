# References
# 1) https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
#    To understand how to split the data into train, test and validation sets

##########################################################################

import csv
import random

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
ccd_tot = [0] * m
fv = [0] * n_feat
ccd_means = []
for i in range (0, m):
    ccd_means.append(fv)

for row in dataset:
    cl = int(row[n_cols-1])
    lists = []
    lists.append(ccd_means[cl])
    lists.append(list(row[0:n_cols-1]))
    ccd_means[cl] = [sum(x) for x in zip(*lists)]
    ccd_tot[cl] = ccd_tot[cl] + 1

for i in range(0, m):
    for j in range (0, n_feat):
     ccd_means[i][j] = ccd_means[i][j] / ccd_tot[i]
print(ccd_means)
