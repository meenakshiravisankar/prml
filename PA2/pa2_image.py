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
    while (iter <= 10000 and not(correctlyClassified(w, X, y))):
        for i in range(X.shape[0]):
            pred = np.matmul(w, X[i])
            if (pred <= 0) and (y[i] == 1):
                w = w + eta*X[i]
            elif (pred >= 0) and (y[i] == 0):
                w = w - eta*X[i]
        iter = iter + 1
    return w

def getConfusion(y, pred, name):
    cnf_matrix = confusion_matrix(y, pred)
    class_names = np.unique(pred, return_counts=False)
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plt.figure()
    cf_mat.plot_confusion_matrix(cnf_matrix, classes=class_names,title=name)
    plt.savefig("results/perceptron/"+name)
    return

# supporting function to return class name corresponding to a class label value
def getName(x):
    if x == 1:
        return "Class Ape"
    return "Class Fan"

# range of learning rates for perceptron algorithm
etas = [0.01, 0.1, 1, 10]

# three different types of initialisation of the weight vector - zero, constant, random
weights = [np.zeros(2048), np.random.rand()*np.ones(2048), np.array(np.random.rand(2048))]

# read datasets
data1 = np.load('../Datasets_PRML_A2/Image_Dataset/fan.npy')
data2 = np.load('../Datasets_PRML_A2/Image_Dataset/ape.npy')

data1 = np.concatenate((data1,np.transpose(np.zeros(1000).reshape(1,1000))),axis = 1)
data2 = np.concatenate((data2,np.transpose(np.ones(1000).reshape(1,1000))),axis = 1)
data = np.concatenate((data1,data2))

# shuffling
np.random.shuffle(data)

# splitting into train, val, test
train_size = int(0.65*0.8*data.shape[0])
val_size = int(0.65*0.2*data.shape[0])
test_size = int(0.35*data.shape[0])

X_train = data[:train_size,:2048]
y_train = data[:train_size,-1]

X_val = data[train_size:train_size+val_size,:2048]
y_val = data[train_size:train_size+val_size,-1]

X_test = data[train_size+val_size:train_size+val_size+test_size,:2048]
y_test = data[train_size+val_size:train_size+val_size+test_size,-1]

acc_train = []
acc_val = []
acc_test = []

# storing the best model parameters (weight initialisation, learning rate eta) and best model validation accuracy
best_model_wt = []
best_model_wt_init_type = 0
best_model_eta = 0
best_val_acc = 0

print("Image Dataset (Fan, Ape)")
print("Number of features = 2048")
print("Size of train, validation and test sets =", X_train.shape, X_val.shape, X_test.shape)
classes, counts = np.unique(y_train, return_counts=True)
print("Number of classes =", len(classes))
print("Counts for each class =", counts)
print("\n")

for w in range(len(weights)):
    print("Initial w =", weights[w])
    for i in range(len(etas)):
        # training
        w_train = runPerceptron(weights[w], X_train, y_train, etas[i])
        acc_train.append(accuracy(w_train, X_train, y_train))

        # validation
        ac = accuracy(w_train, X_val, y_val)
        acc_val.append(ac)
        if ac > best_val_acc:
            best_val_acc = ac
            best_model_wt = w_train
            best_model_wt_init_type = w
            best_model_eta = etas[i]

        # test
        acc_test.append(accuracy(w_train, X_test, y_test))

np.savetxt('results/perceptron/Train_acc_image_ds',acc_train,fmt='%.2f')
np.savetxt('results/perceptron/Val_acc_image_ds',acc_val,fmt='%.2f')
np.savetxt('results/perceptron/Test_acc_image_ds',acc_test,fmt='%.2f')

print("Best model for Image Dataset (Fan, Ape):")
print("Weight initialisation type:", best_model_wt_init_type)
print("Weights:", best_model_wt)
print("Learning rate:", best_model_eta)
print("Validation Accuracy:", best_val_acc)
getConfusion(y_test, predictions(best_model_wt, X_test), "image_ds_best_model")
