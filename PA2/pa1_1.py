import numpy as np
import matplotlib.pyplot as plt


# calculate the accuracy of classification
def get_accuracy(pred, y):
    return np.sum(pred == y)/len(y)*100

def get_sigmoid(z) :
    return 1/(1+np.exp(-z))

def get_wt_init(wt_init,size) :
    if wt_init == 0 :
        w = np.zeros((size,1))
    elif wt_init == 1 :
        w = np.ones((size,1))
    elif wt_init == 2 :
        w = np.random.uniform(low=0,high=1,size=(size,1))
    return w

# def get_norm(data) :
    # data

def get_description(data) :
    print("Mean of train data ",np.mean(data,axis=0))
    print("Std of train data",np.std(data,axis=0))
# read dataset
data = np.loadtxt("../Datasets_PRML_A2/Dataset_4_Team_39.csv", delimiter=',', dtype=None)
classes = np.array(np.unique(data[:,2], return_counts=False),dtype=int)

# shuffling
np.random.shuffle(data)
# normalize data
# data = get_norm(data)

# splitting into train, test as 65% and 35%
train_size = int(0.65*data.shape[0])
test_size = int(0.35*data.shape[0])
# splitting into train and val sets from total train data
val_size = int(0.2*train_size)
train_size = int(0.8*train_size)

X_train = data[:train_size,:2]
y_train = data[:train_size,-1].reshape(-1,1)
X_val = data[train_size:train_size+val_size,:2]
y_val = data[train_size:train_size+val_size,-1].reshape(-1,1)
X_test = data[train_size+val_size:train_size+val_size+test_size,:2]
y_test = data[train_size+val_size:train_size+val_size+test_size,-1].reshape(-1,1)

get_description(X_train)

print("Size of train, validation and test sets -",X_train.shape,X_val.shape,X_test.shape)
print("Classes -",classes)

# hyperparameters
wt_inits = [0,1,2] # 0-zero weights, 1-non-zero constant weights, 2-random weights
lrs = [0.001,0.01,0.1,1,10]
iterations = [2,1000,10000]

default = 1

if default :
    wt_inits = [wt_inits[0]]
    lrs = [lrs[0]]
    iterations = [iterations[0]]
    
# Initialisation of weights
size = X_train.shape[1]

train_accuracies = []
val_accuracies = []

for wt_init in wt_inits :
    for iteration in iterations :
        for lr in lrs :
            w = get_wt_init(wt_init,size)
            for i in range(iteration) :
                # Compute prediction based on current weight
                y_train_pred = get_sigmoid(np.matmul(X_train,w))
                # compute gradient of cross-entropy loss for binary classification
                grad_err = np.matmul( np.transpose(X_train),y_train_pred-y_train)
                # accuracy of current prediction
                y_train_pred[y_train_pred>=0.5] = 1
                y_train_pred[y_train_pred<0.5] = 0
                train_acc = get_accuracy(y_train_pred,y_train)
                # accuracy on validation set
                val_pred = get_sigmoid(np.matmul(X_val,w))
                val_pred[val_pred>=0.5] = 1
                val_pred[val_pred<0.5] = 0
                train_acc = get_accuracy(val_pred,y_train)
                val_acc = get_accuracy(val_pred,y_val)
                # Update rule
                w -= (lr*grad_err)
                print("Train accuracy {:.2f}".format(train_acc))
                print("Validation accuracy {:.2f}".format(val_acc))
        


