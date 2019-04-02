# Reference: https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/

import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
from sklearn.metrics import confusion_matrix
import confusion_matrix as cf_mat
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# seeding
np.random.seed(seed=42)

# compute the confusion matrix
def getConfusion(y, pred, name) :
    cnf_matrix = confusion_matrix(y, pred)
    class_names = np.unique(pred, return_counts=False)
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plt.figure()
    cf_mat.plot_confusion_matrix(cnf_matrix, classes=class_names,title=name)
    plt.savefig("results/"+name)
    return

# calculate the accuracy of classification
def accuracy(w, X, y):
    predvals = predictions(w, X)
    preds = (predvals == y)
    corrClass = sum(preds)
    acc = (100.0*corrClass)/(X.shape[0]+0.0)
    return acc

# reading the dataset
data1 = np.loadtxt("../Datasets_PRML_A2/Dataset_4_Team_39.csv", delimiter=',', dtype=None)

# shuffling the data
np.random.shuffle(data1)

# splitting into train, test - 65, 35
train_size = int(0.65*data1.shape[0])
test_size = int(0.35*data1.shape[0])

X_train = data1[:train_size,:2]
y_train = data1[:train_size,-1]

X_test = data1[train_size:train_size+test_size,:2]
y_test = data1[train_size:train_size+test_size,-1]

# range of values for non-linear SVM classification error parameter 'C'
C_vals = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
for i in range(len(C_vals)):
    # training an sklearn SVM using the data
    svclassifier = SVC(C=C_vals[i], kernel='linear')
    svclassifier.fit(X_train, y_train)

    # getting the predictions
    y_pred = svclassifier.predict(X_test)

    # evaluating the classification
    getConfusion(y_test, y_pred, "SVM_Dataset4_"+str(i))

# reading the dataset
data2 = np.loadtxt("../Datasets_PRML_A2/Dataset_5_Team_39.csv", delimiter=',', dtype=None)

# shuffling the data
np.random.shuffle(data2)

# splitting into train, test - 65, 35
train_size = int(0.65*data2.shape[0])
test_size = int(0.35*data2.shape[0])

X_train = data2[:train_size,:2]
y_train = data2[:train_size,-1]

X_test = data2[train_size:train_size+test_size,:2]
y_test = data2[train_size:train_size+test_size,-1]

# range of values for non-linear SVM classification error parameter 'C'
C_vals = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
for i in range(len(C_vals)):
    # training an sklearn SVM using the data
    svclassifier = SVC(C=C_vals[i], kernel='linear')
    svclassifier.fit(X_train, y_train)

    # getting the predictions
    y_pred = svclassifier.predict(X_test)

    # evaluating the classification
    getConfusion(y_test, y_pred, "SVM_Dataset5_"+str(i))
