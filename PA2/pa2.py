import numpy as np
from sklearn.metrics import confusion_matrix
import confusion_matrix as cf_mat
import matplotlib.pyplot as plt

def correctlyClassified(w, X, y):
    for i in range(X.shape[0]):
        pred = np.matmul(w, X[i])
        if (pred <= 0) and (y[i] == 1):
            return False
        elif (pred >= 0) and (y[i] == 0):
            return False
    return True

def predictions(w, X):
    preds = []
    for i in range(X.shape[0]):
        pred = np.matmul(w, X[i])
        if (pred < 0):
            preds.append(0)
        elif (pred > 0):
            preds.append(1)
    return np.array(preds)

def costFunction(w, X, y):
    cost = 0
    for i in range(X.shape[0]):
        pred = np.matmul(w, X[i])
        if y[i] == 0:
            cost = cost - pred
        else:
            cost = cost + pred
    return (-1 * cost)

def runPerceptron(w, X, y, eta):
    iter = 0
    while (iter <= 10000 and not(correctlyClassified(w, X, y))):
        for i in range(X.shape[0]):
            pred = np.matmul(w, X[i])
            if (pred <= 0) and (y[i] == 1):
                w = w + eta*X[i]
            elif (pred >= 0) and (y[i] == 0):
                w = w - eta*X[i]
        iter = iter + 1
    return w

# Compute confusion matrix
# change this to test data
def getConfusion(y_train, prediction, name) :
    # confusion matrix for test - change this...
    cnf_matrix = confusion_matrix(y_train, prediction)
    class_names = np.unique(prediction, return_counts=False)
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plt.figure()
    cf_mat.plot_confusion_matrix(cnf_matrix, classes=class_names,title=name)
    plt.savefig("results/"+name)
    # plt.show()
    return

np.random.seed(seed=42)

# read datasets
data1 = np.loadtxt("../Datasets_PRML_A2/Dataset_2_Team_39.csv", delimiter=',', dtype=None)

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
    w_trained = runPerceptron(w_init, X_train, y_train, etas[i])
    print(etas[i], w_trained, costFunction(w_trained, X_train, y_train))
    getConfusion(y_train, predictions(w_trained, X_train), "Dataset 2 Perceptron "+str(i))
