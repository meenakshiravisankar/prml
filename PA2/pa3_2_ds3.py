# References:
# https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/
# https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html
# https://stackoverflow.com/questions/26558816/matplotlib-scatter-plot-with-legend

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import seaborn as sns

# seeding
np.random.seed(seed=42)

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
C_vals = [0.01, 0.1, 1.0, 10.0]

# range of values for degrees for polynomial kernel SVM
degrees = [2, 4, 6, 8]

# range of values for gamma for rbf kernel SVM
gammas = [0.01, 0.1, 1, 10]

# reading the dataset
data = np.loadtxt("../Datasets_PRML_A2/Dataset_3_Team_39.csv", delimiter=',', dtype=None)

# shuffling the data
np.random.shuffle(data)

# splitting into train, val, test
train_size = int(0.65*0.8*data.shape[0])
val_size = int(0.65*0.2*data.shape[0])
test_size = int(0.35*data.shape[0])

X_train = data[:train_size,:2]
y_train = data[:train_size,-1]

X_val = data[train_size:train_size+val_size,:2]
y_val = data[train_size:train_size+val_size,-1]

X_test = data[train_size+val_size:train_size+val_size+test_size,:2]
y_test = data[train_size+val_size:train_size+val_size+test_size,-1]

acc_linear_train = []
acc_linear_val = []
acc_linear_test = []

# storing the best model parameters (C) and best model validation accuracy for linear kernel
best_model_C_1 = 0
best_val_acc_1 = 0
best_svclassifier_1 = None
best_sv_1 = None

# for linear kernel
for i in range(len(C_vals)):
    # training an sklearn SVM using the data
    svclassifier = SVC(C=C_vals[i], kernel='linear')
    svclassifier.fit(X_train, y_train)
    sv = svclassifier.support_vectors_

    # finding the train accuracies
    y_pred = svclassifier.predict(X_train)
    acc_linear_train.append(accuracy(y_pred, y_train))

    # finding the validation accuracies
    y_pred_v = svclassifier.predict(X_val)
    ac = accuracy(y_pred_v, y_val)
    acc_linear_val.append(ac)

    if (ac > best_val_acc_1):
        best_val_acc_1 = ac
        best_model_C_1 = C_vals[i]
        best_svclassifier_1 = svclassifier
        best_sv_1 = sv

    # finding the test accuracies
    y_pred_t = svclassifier.predict(X_test)
    acc_linear_test.append(accuracy(y_pred_t, y_test))

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
Z = best_svclassifier_1.decision_function(xy).reshape(XX.shape)

# plotting decision boundary and margins
axes.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

# plotting support vectors
axes.scatter(best_sv_1[:, 0], best_sv_1[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("results/svmlinker/DS3_boundary_linear")
plt.clf()

print("Best linear kernel SVM model for Dataset 3:")
print("C:", best_model_C_1)
print("Validation Accuracy:", best_val_acc_1)

np.savetxt('results/svmlinker/Train_acc_ds3',acc_linear_train,fmt='%.2f')
np.savetxt('results/svmlinker/Val_acc_ds3',acc_linear_val,fmt='%.2f')
np.savetxt('results/svmlinker/Test_acc_ds3',acc_linear_test,fmt='%.2f')

acc_poly_train = []
acc_poly_val = []
acc_poly_test = []

# storing the best model parameters (C, degree) and best model validation accuracy for polynomial kernel
best_model_C_2 = 0
best_model_deg_2 = 0
best_val_acc_2 = 0
best_svclassifier_2 = None
best_sv_2 = None

# for polynomial kernel
for i in range(len(degrees)):
    for j in range(len(C_vals)):
        # training an sklearn SVM using the data
        svclassifier = SVC(C=C_vals[j], kernel='poly', coef0=1, degree=degrees[i])
        svclassifier.fit(X_train, y_train)
        sv = svclassifier.support_vectors_

        # finding the train accuracies
        y_pred = svclassifier.predict(X_train)
        acc_poly_train.append(accuracy(y_pred, y_train))

        # finding the validation accuracies
        y_pred_v = svclassifier.predict(X_val)
        ac = accuracy(y_pred_v, y_val)
        acc_poly_val.append(ac)

        if (ac > best_val_acc_2):
            best_val_acc_2 = ac
            best_model_C_2 = C_vals[j]
            best_model_deg_2 = degrees[i]
            best_svclassifier_2 = svclassifier
            best_sv_2 = sv

        # finding the test accuracies
        y_pred_t = svclassifier.predict(X_test)
        acc_poly_test.append(accuracy(y_pred_t, y_test))

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
Z = best_svclassifier_2.decision_function(xy).reshape(XX.shape)

# plotting decision boundary and margins
axes.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

# plotting support vectors
axes.scatter(best_sv_2[:, 0], best_sv_2[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("results/svmpolyker/DS3_boundary_poly")
plt.clf()

print("Best polynomial kernel SVM model for Dataset 3:")
print("C:", best_model_C_2)
print("Degree:", best_model_deg_2)
print("Validation Accuracy:", best_val_acc_2)

np.savetxt('results/svmpolyker/Train_acc_ds3',acc_poly_train,fmt='%.2f')
np.savetxt('results/svmpolyker/Val_acc_ds3',acc_poly_val,fmt='%.2f')
np.savetxt('results/svmpolyker/Test_acc_ds3',acc_poly_test,fmt='%.2f')

acc_rbf_train = []
acc_rbf_val = []
acc_rbf_test = []

# storing the best model parameters (C) and best model validation accuracy for rbf kernel
best_model_C_3 = 0
best_model_gamma_3 = 0
best_val_acc_3 = 0
best_svclassifier_3 = None
best_sv_3 = None

# for rbf kernel
for i in range(len(gammas)):
    for j in range(len(C_vals)):
        # training an sklearn SVM using the data
        svclassifier = SVC(C=C_vals[j], kernel='rbf', gamma=gammas[i])
        svclassifier.fit(X_train, y_train)
        sv = svclassifier.support_vectors_

        # finding the train accuracies
        y_pred = svclassifier.predict(X_train)
        acc_rbf_train.append(accuracy(y_pred, y_train))

        # finding the validation accuracies
        y_pred_v = svclassifier.predict(X_val)
        ac = accuracy(y_pred_v, y_val)
        acc_rbf_val.append(ac)

        if (ac > best_val_acc_3):
            best_val_acc_3 = ac
            best_model_C_3 = C_vals[j]
            best_model_gamma_3 = gammas[i]
            best_svclassifier_3 = svclassifier
            best_sv_3 = sv

        # finding the test accuracies
        y_pred_t = svclassifier.predict(X_test)
        acc_rbf_test.append(accuracy(y_pred_t, y_test))

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
Z = best_svclassifier_3.decision_function(xy).reshape(XX.shape)

# plotting decision boundary and margins
axes.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

# plotting support vectors
axes.scatter(best_sv_3[:, 0], best_sv_3[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("results/svmrbfker/DS3_boundary_rbf")
plt.clf()

print("Best rbf kernel SVM model for Dataset 3:")
print("C:", best_model_C_3)
print("Gamma:", best_model_gamma_3)
print("Validation Accuracy:", best_val_acc_3)

np.savetxt('results/svmrbfker/Train_acc_ds3',acc_rbf_train,fmt='%.2f')
np.savetxt('results/svmrbfker/Val_acc_ds3',acc_rbf_val,fmt='%.2f')
np.savetxt('results/svmrbfker/Test_acc_ds3',acc_rbf_test,fmt='%.2f')
