import numpy as np
from sklearn.metrics import confusion_matrix
import confusion_matrix as cf_mat
import matplotlib.pyplot as plt
from random import randint

np.random.seed(seed=42)

def correctlyClassified(w, X, y):
    predvals = predictions(w, X)
    preds = (predvals == y)
    return preds.all()

def predictions(w, X):
    preds = []
    for i in range(X.shape[0]):
        pred = np.matmul(w, X[i])
        if (pred < 0):
            preds.append(0)
        else:
            preds.append(1)
    return np.array(preds)

def accuracy(w, X, y):
    predvals = predictions(w, X)
    preds = (predvals == y)
    corrClass = sum(preds)
    acc = (100.0*corrClass)/(X.shape[0]+0.0)
    return acc

def runPerceptron(w, X, y, eta):
    iter = 0
    acc = []
    while (iter <= 10000 and not(correctlyClassified(w, X, y))):
        for i in range(X.shape[0]):
            pred = np.matmul(w, X[i])
            if (pred <= 0) and (y[i] == 1):
                w = w + eta*X[i]
            elif (pred >= 0) and (y[i] == 0):
                w = w - eta*X[i]
        iter = iter + 1
        if (iter%100 == 0):
            acc.append(accuracy(w, X, y))
    print("iterations taken =", iter)
    return w, acc

def getConfusion(y, pred, name):
    cnf_matrix = confusion_matrix(y, pred)
    class_names = np.unique(pred, return_counts=False)
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plt.figure()
    cf_mat.plot_confusion_matrix(cnf_matrix, classes=class_names,title=name)
    plt.savefig("results/perceptron/"+name)
    return

def makeAccVsIterPlot(acc, name):
    iter = [x*100 for x in range(1, len(acc)+1)]
    plt.figure()
    plt.plot(iter, acc)
    plt.savefig("results/perceptron/"+name)
    return

# range of learning rates for perceptron algorithm
etas = [0.01, 0.1, 1, 10]

# three different types of initialisation of the weight vector - zero, constant, random
weights = [np.array([0, 0]), np.random.rand()*np.array([1, 1]), np.array(np.random.rand(2))]

# # read dataset
# data1 = np.loadtxt("../Datasets_PRML_A2/Dataset_2_Team_39.csv", delimiter=',', dtype=None)
#
# # shuffling
# np.random.shuffle(data1)
#
# # splitting into train, val, test
# train_size = int(0.65*0.8*data1.shape[0])
# val_size = int(0.65*0.2*data1.shape[0])
# test_size = int(0.35*data1.shape[0])
#
# X_train = data1[:train_size,:2]
# y_train = data1[:train_size,-1]
#
# X_val = data1[train_size:train_size+val_size,:2]
# y_val = data1[train_size:train_size+val_size,-1]
#
# X_test = data1[train_size+val_size:train_size+val_size+test_size,:2]
# y_test = data1[train_size+val_size:train_size+val_size+test_size,-1]
#
# acc_train_data1 = []
# acc_val_data1 = []
# acc_test_data1 = []
#
# print("Dataset 2")
# print("Number of features = 2")
# print("Size of train, validation and test sets =", X_train.shape, X_val.shape, X_test.shape)
# classes, counts = np.unique(y_train, return_counts=True)
# print("Number of classes =", len(classes))
# print("Counts for each class =", counts)
# print("\n")
#
# for w in range(len(weights)):
#     print("Initial w =", weights[w])
#     for i in range(len(etas)):
#         # training
#         w_train, acc1 = runPerceptron(weights[w], X_train, y_train, etas[i])
#         # makeAccVsIterPlot(acc1, "Iteration_vs_acc_ds2_wt_"+str(w)+"_eta_"+str(i))
#         acc_train_data1.append(accuracy(w_train, X_train, y_train))
#
#         # validation
#         acc_val_data1.append(accuracy(w_train, X_val, y_val))
#
#         # test
#         acc_test_data1.append(accuracy(w_train, X_test, y_test))
#
#         getConfusion(y_test, predictions(w_train, X_test), "DS2_wt_"+str(w)+"_eta_"+str(i))
#
# np.savetxt('results/perceptron/Train_acc_ds2',acc_train_data1,fmt='%.2f')
# np.savetxt('results/perceptron/Val_acc_ds2',acc_val_data1,fmt='%.2f')
# np.savetxt('results/perceptron/Test_acc_ds2',acc_test_data1,fmt='%.2f')

print("\n************************\n")

# read dataset
data2 = np.loadtxt("../Datasets_PRML_A2/Dataset_4_Team_39.csv", delimiter=',', dtype=None)

# shuffling
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

print("Dataset 4")
print("Number of features = 2")
print("Size of train, validation and test sets =", X_train.shape, X_val.shape, X_test.shape)
classes, counts = np.unique(y_train, return_counts=True)
print("Number of classes =", len(classes))
print("Counts for each class =", counts)
print("\n")

for w in range(len(weights)):
    print("Initial w =", weights[w])
    for i in range(len(etas)):
        # training
        w_train, acc2 = runPerceptron(weights[w], X_train, y_train, etas[i])
        # makeAccVsIterPlot(acc1, "Iteration_vs_acc_ds2_wt_"+str(w)+"_eta_"+str(i))
        acc_train_data2.append(accuracy(w_train, X_train, y_train))

        # validation
        acc_val_data2.append(accuracy(w_train, X_val, y_val))

        # test
        acc_test_data2.append(accuracy(w_train, X_test, y_test))

        getConfusion(y_test, predictions(w_train, X_test), "DS4_wt_"+str(w)+"_eta_"+str(i))

np.savetxt('results/perceptron/Train_acc_ds4',acc_train_data2,fmt='%.2f')
np.savetxt('results/perceptron/Val_acc_ds4',acc_val_data2,fmt='%.2f')
np.savetxt('results/perceptron/Test_acc_ds4',acc_test_data2,fmt='%.2f')
