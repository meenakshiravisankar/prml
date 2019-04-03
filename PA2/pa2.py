import numpy as np
from sklearn.metrics import confusion_matrix
import sklearn.datasets
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
        elif (pred > 0):
            preds.append(1)
    return np.array(preds)

# check if this is useful or not necessary
def errFunction(w, X, y):
    err = 0
    predvals = predictions(w, X)
    preds = (predvals == y)
    for i in range(len(preds)):
        if not(preds[i]):
            if y[i] == 0:
                err = err + predvals[i]
            else:
                err = err - predvals[i]
    return err

def accuracy(w, X, y):
    predvals = predictions(w, X)
    preds = (predvals == y)
    corrClass = sum(preds)
    acc = (100.0*corrClass)/(X.shape[0]+0.0)
    return acc

def runPerceptron(w, X, y, eta):
    iter = 0
    # check this
    while (iter <= 10000 and not(correctlyClassified(w, X, y))):
        for i in range(X.shape[0]):
            pred = np.matmul(w, X[i])
            if (pred <= 0) and (y[i] == 1):
                w = w + eta*X[i]
            elif (pred >= 0) and (y[i] == 0):
                w = w - eta*X[i]
        iter = iter + 1
    print("iterations taken =", iter)
    return w

def getConfusion(y, pred, name) :
    cnf_matrix = confusion_matrix(y, pred)
    class_names = np.unique(pred, return_counts=False)
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plt.figure()
    cf_mat.plot_confusion_matrix(cnf_matrix, classes=class_names,title=name)
    plt.savefig("results/"+name)
    # plt.show()
    return

# X, y = sklearn.datasets.make_classification(n_samples=30, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, class_sep = 3, n_classes=2)
# f = plt.figure(1)
# plt.scatter(X[:,0],X[:,1],c=y)
# plt.savefig("results/dummy_scatter")
#
# w_init = np.array(np.random.rand(2))
# etas = [0.01, 0.1, 1, 10]
# for i in range(len(etas)):
#     w = runPerceptron(w_init, X, y, etas[i])
#     print(etas[i], w, accuracy(w, X, y))
#     getConfusion(y, predictions(w, X), "Dummy Perceptron "+str(i))

# read datasets
data1 = np.loadtxt("../Datasets_PRML_A2/Dataset_2_Team_39.csv", delimiter=',', dtype=None)

data2 = np.loadtxt("../Datasets_PRML_A2/Dataset_4_Team_39.csv", delimiter=',', dtype=None)

# shuffling
np.random.shuffle(data1)

# splitting into train, test - 65, 35
train_size = int(0.65*data1.shape[0])
test_size = int(0.35*data1.shape[0])

X_train = data1[:train_size,:2]
y_train = data1[:train_size,-1]

X_test = data1[train_size:train_size+test_size,:2]
y_test = data1[train_size:train_size+test_size,-1]

print("Dataset 2")
print("Number of features = 2")
print("Size of train and test sets =", X_train.shape, X_test.shape)
classes, counts = np.unique(y_train, return_counts=True)
print("Number of classes =", len(classes))
print("Counts for each class =", counts)
print("\n")

# random initial choice of w
w_init = np.array(np.random.rand(2))
print("Initial w =", w_init)

etas = [0.01, 0.1, 1, 10]
for i in range(len(etas)):
    w_train = runPerceptron(w_init, X_train, y_train, etas[i])
    print(etas[i], w_train, accuracy(w_train, X_train, y_train))
    getConfusion(y_train, predictions(w_train, X_train), "Dataset2_Perceptron_"+str(i))
    # w_test = runPerceptron(w_init, X_test, y_test, etas[i])
    # print(etas[i], w_test, accuracy(w_test, X_test, y_test))
    # getConfusion(y_test, predictions(w_test, X_test), "Dataset2_Perceptron_"+str(i))

# shuffling
np.random.shuffle(data2)

# splitting into train, test - 65, 35
train_size = int(0.65*data2.shape[0])
test_size = int(0.35*data2.shape[0])

X_train = data2[:train_size,:2]
y_train = data2[:train_size,-1]

X_test = data2[train_size:train_size+test_size,:2]
y_test = data2[train_size:train_size+test_size,-1]

print("Dataset 4")
print("Number of features = 2")
print("Size of train and test sets =", X_train.shape, X_test.shape)
classes, counts = np.unique(y_train, return_counts=True)
print("Number of classes =", len(classes))
print("Counts for each class =", counts)
print("\n")

# random initial choice of w
w_init = np.array(np.random.rand(2))
print("Initial w =", w_init)

etas = [0.01, 0.1, 1, 10]
for i in range(len(etas)):
    w_train = runPerceptron(w_init, X_train, y_train, etas[i])
    print(etas[i], w_train, accuracy(w_train, X_train, y_train))
    getConfusion(y_train, predictions(w_train, X_train), "Dataset4_Perceptron_"+str(i))
    # w_test = runPerceptron(w_init, X_test, y_test, etas[i])
    # print(etas[i], w_test, accuracy(w_test, X_test, y_test))
    # getConfusion(y_test, predictions(w_test, X_test), "Dataset4_Perceptron_"+str(i))
