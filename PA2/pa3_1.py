# References:
# https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/
# https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html
# https://stackoverflow.com/questions/26558816/matplotlib-scatter-plot-with-legend

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import confusion_matrix as cf_mat
from sklearn.svm import SVC
import seaborn as sns

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
    plt.savefig("results/svm/"+name)
    return

# calculate the accuracy of classification
def accuracy(pred, y):
    result = (pred == y)
    corrClass = sum(result)
    acc = (100.0*corrClass)/(len(y)+0.0)
    return acc

# supporting function to return class name corresponding to a class label value
def getName(x):
    if x == 1:
        return "Class 1"
    return "Class 0"

# class labels
classes = [0, 1]

# range of values for non-linear SVM classification error parameter 'C'
C_vals = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

# reading the dataset
data1 = np.loadtxt("../Datasets_PRML_A2/Dataset_4_Team_39.csv", delimiter=',', dtype=None)

# shuffling the data
np.random.shuffle(data1)

# splitting into train, val, test
train_size = int(0.65*0.8*data1.shape[0])
val_size = int(0.65*0.2*data1.shape[0])
test_size = int(0.35*data1.shape[0])

X_train = data1[:train_size,:2]
y_train = data1[:train_size,-1]

X_val = data1[train_size:train_size+val_size,:2]
y_val = data1[train_size:train_size+val_size,-1]

X_test = data1[train_size+val_size:train_size+val_size+test_size,:2]
y_test = data1[train_size+val_size:train_size+val_size+test_size,-1]

acc_train_data1 = []
acc_val_data1 = []
acc_test_data1 = []

for i in range(len(C_vals)):
    # training an sklearn SVM using the data
    svclassifier = SVC(C=C_vals[i], kernel='linear')
    svclassifier.fit(X_train, y_train)
    sv = svclassifier.support_vectors_

    # finding the train accuracies
    y_pred = svclassifier.predict(X_train)
    acc_train_data1.append(accuracy(y_pred, y_train))

    # finding the validation accuracies
    y_pred_v = svclassifier.predict(X_val)
    acc_val_data1.append(accuracy(y_pred_v, y_val))

    # finding the test accuracies
    y_pred_t = svclassifier.predict(X_test)
    acc_test_data1.append(accuracy(y_pred_t, y_test))

    f = plt.figure()
    sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=list(map(getName,y_train)))

    # getting the axes information of the plot
    axes = plt.gca()
    xlim = axes.get_xlim()
    ylim = axes.get_ylim()

    # creating a grid to help plot the decision function
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.transpose(np.vstack([XX.ravel(), YY.ravel()]))
    Z = svclassifier.decision_function(xy).reshape(XX.shape)

    # plotting decision boundary and margins
    axes.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

    # plotting support vectors
    axes.scatter(sv[:, 0], sv[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.savefig("results/svm/DS4_boundary"+str(i))

    # evaluating the confusion matrix
    getConfusion(y_test, y_pred_t, "DS4_"+str(i))

np.savetxt('results/svm/Train_acc_ds4',acc_train_data1,fmt='%.2f')
np.savetxt('results/svm/Val_acc_ds4',acc_val_data1,fmt='%.2f')
np.savetxt('results/svm/Test_acc_ds4',acc_test_data1,fmt='%.2f')

# reading the dataset
data2 = np.loadtxt("../Datasets_PRML_A2/Dataset_5_Team_39.csv", delimiter=',', dtype=None)

# shuffling the data
np.random.shuffle(data2)

# splitting into train, val, test
train_size = int(0.65*0.8*data2.shape[0])
val_size = int(0.65*0.2*data2.shape[0])
test_size = int(0.35*data2.shape[0])

X_train = data2[:train_size,:2]
y_train = data2[:train_size,-1]

X_val = data2[train_size:train_size+val_size,:2]
y_val = data2[train_size:train_size+val_size,-1]

X_test = data2[train_size+val_size:train_size+val_size+test_size,:2]
y_test = data2[train_size+val_size:train_size+val_size+test_size,-1]

acc_train_data2 = []
acc_val_data2 = []
acc_test_data2 = []

for i in range(len(C_vals)):
    # training an sklearn SVM using the data
    svclassifier = SVC(C=C_vals[i], kernel='linear')
    clf = svclassifier.fit(X_train, y_train)
    sv = svclassifier.support_vectors_

    # finding the train accuracies
    y_pred = svclassifier.predict(X_train)
    acc_train_data2.append(accuracy(y_pred, y_train))

    # finding the validation accuracies
    y_pred_v = svclassifier.predict(X_val)
    acc_val_data2.append(accuracy(y_pred_v, y_val))

    # finding the test accuracies
    y_pred_t = svclassifier.predict(X_test)
    acc_test_data2.append(accuracy(y_pred_t, y_test))

    f = plt.figure()
    sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=list(map(getName,y_train)))

    # getting the axes information of the plot
    axes = plt.gca()
    xlim = axes.get_xlim()
    ylim = axes.get_ylim()

    # creating a grid to help plot the decision function
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.transpose(np.vstack([XX.ravel(), YY.ravel()]))
    Z = svclassifier.decision_function(xy).reshape(XX.shape)

    # plotting decision boundary and margins
    axes.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

    # plotting support vectors
    axes.scatter(sv[:, 0], sv[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.savefig("results/svm/DS5_boundary"+str(i))

    # evaluating the confusion matrix
    getConfusion(y_test, y_pred_t, "DS5_"+str(i))

np.savetxt('results/svm/Train_acc_ds5',acc_train_data2,fmt='%.2f')
np.savetxt('results/svm/Val_acc_ds5',acc_val_data2,fmt='%.2f')
np.savetxt('results/svm/Test_acc_ds5',acc_test_data2,fmt='%.2f')
