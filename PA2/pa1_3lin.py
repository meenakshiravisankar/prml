import numpy as np
import matplotlib.pyplot as plt
import warnings
import confusion_matrix as cf_mat
from sklearn.metrics import confusion_matrix

# calculate the accuracy of classification
def get_accuracy(pred, y):
    return np.sum(pred == y)/len(y)*100

def get_sigmoid(z) :
    sig = 1/(1+np.exp(-z))
    return sig

def get_wt_init(wt_init,size) :
    if wt_init == 0 :
        w = np.zeros((size,1))
    elif wt_init == 1 :
        w = np.ones((size,1))
    elif wt_init == 2 :
        w = np.random.uniform(low=0,high=1,size=(size,1))
    elif wt_init == 3 :
        w = np.random.standard_normal((size,1))
    return w

def get_standardization(data,mean,std,standard) :
    if standard :
        return np.divide(data-mean,std)
    else :
        return data

def get_description(data) :
    mean = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    return mean, std

def get_class(data) :
    data[data>=0.5] = 1
    data[data<0.5] = 0
    return data

# Compute confusion matrix
def getConfusion(y_test, prediction, name, title) :

    # confusion matrix for test
    cnf_matrix = confusion_matrix(y_test, prediction)
    class_names = np.unique(prediction, return_counts=False)
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plt.figure()
    cf_mat.plot_confusion_matrix(cnf_matrix, classes=class_names)
    plt.title(title)
    plt.savefig("results/"+name)
    # plt.show()
    return

np.random.seed(seed=42)

dataset = "4"
# read dataset
data = np.loadtxt("../Datasets_PRML_A2/Dataset_"+dataset+"_Team_39.csv", delimiter=',', dtype=None)
classes = np.array(np.unique(data[:,2], return_counts=False),dtype=int)

# shuffling
np.random.shuffle(data)
warnings.simplefilter("ignore")

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

# compute mean and std of train data
mini = np.min(X_train, axis=0)
maxi = np.max(X_train, axis=0)
mean,std = get_description(X_train)

print("Logistic Regression")
print("Size of train, validation and test sets -",X_train.shape,X_val.shape,X_test.shape)
print("Classes -",classes)

standard = 1
iterations = [100,1000,10000]
boundary_plot = 0

# hyperparameters 1
if standard : 
    wt_inits = [0,1,2] # 0-zero weights, 1-non-zero constant weights, 2-random weights
    lrs = [10**-4,0.001,0.01,0.1,1]
    iteration = 1000
    weight_init = ["Zero", "Constant", "Random uniform"]
    word = ""
else :
    wt_inits = [0,1,2] # 0-zero weights, 1-non-zero constant weights, 2-random weights
    lrs = [10**-6,10**-5,0.0001]
    iteration = 10000
    weight_init = ["Zero", "Constant", "Random uniform"]
    word = ""

default = 0

if default :
    wt_inits = [wt_inits[0]]
    lrs = [lrs[0]]
    iterations = [iterations[0]]

configs = []

# Initialisation of weights
size = X_train.shape[1]

train_accuracies = []
val_accuracies = []

total_iterations = [x for x in range(iteration)]

X_train = get_standardization(X_train, mean, std, standard)
X_val = get_standardization(X_val, mean, std, standard)
X_test = get_standardization(X_test, mean, std, standard)

figs = 221

for wt_init in wt_inits :
    # plt.figure()
    for lr in lrs :
        w = get_wt_init(wt_init,size)
        iteration_training = []
        for i in range(iteration) :
            # Compute prediction based on current weight
            y_train_pred = get_sigmoid(np.matmul(X_train,w))
            # compute gradient of cross-entropy loss for binary classification
            grad_err = np.matmul( np.transpose(X_train),y_train_pred-y_train)
            # accuracy of current prediction
            train_acc = get_accuracy(get_class(y_train_pred),y_train)
            # prediction on validation set
            y_val_pred = get_sigmoid(np.matmul(X_val,w))
            # accuracy on validation set
            val_acc = get_accuracy(get_class(y_val_pred),y_val)
            # Update rule
            w -= (lr*grad_err)
            # Accuracy values
            # print("Train accuracy {:.2f}".format(train_acc))
            # print("Validation accuracy {:.2f}".format(val_acc))
            iteration_training.append(train_acc)
        
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        configs.append([wt_init, lr])
        
        plt.subplot(figs)
        plt.plot(total_iterations,iteration_training,label="lr="+str(lr))
    
    figs+=1
    # plt.axis([0,iteration,0,100])
    plt.title(weight_init[wt_init]+" weight initialization",size=8)
    plt.axis(size=8)
    if figs == 222 :
        plt.ylabel("Accuracy",size=8)
    if figs == 225 :
        plt.xlabel("Iterations",size=8)
    plt.tight_layout()
    plt.legend()
    # plt.savefig("results/kernel/linear/withoutstdds"+dataset+""+weight_init[wt_init])
plt.savefig("results/logreg/kernel/linear/"+word+"ds"+dataset+"accuracy")
idx = np.argmax(np.array(val_accuracies))
best_model = configs[idx]
print("Best model has validation accuracy {:.2f}".format(np.max(np.array(val_accuracies))))

wt_init = best_model[0]
lr = best_model[1]
results = []
# Saving confusion matrix on test, accuracy on train and test data for best model
w = get_wt_init(wt_init,size)
for i in range(iteration) :
    # Compute prediction based on current weight
    y_train_pred = get_sigmoid(np.matmul(X_train,w))
    # compute gradient of cross-entropy loss for binary classification
    grad_err = np.matmul( np.transpose(X_train),y_train_pred-y_train)
    # Update rule
    w -= (lr*grad_err)

pred = get_class(get_sigmoid(np.matmul(X_train,w)))
acc = get_accuracy(pred,y_train)
results.append(acc)
pred = get_sigmoid(np.matmul(X_test,w))
acc = get_accuracy(get_class(pred),y_test)
results.append(acc)
getConfusion(y_test, pred, "logreg/kernel/linear/"+word+"ds"+dataset+"cfmatrix", "Dataset "+dataset+" - Test data")
np.savetxt("results/logreg/kernel/linear/"+word+"ds"+dataset+"traintest.txt",results,fmt="%.2f")


if boundary_plot :
    X_train = data[:train_size,:2]
    y_train = data[:train_size,-1]

    fig = plt.figure()

    # Train scatter plot
    plt.plot(X_train[y_train==0][:,0],X_train[y_train==0][:,1],'.',c='r',label="class 0")
    plt.plot(X_train[y_train==1][:,0],X_train[y_train==1][:,1],'.',c='g',label="class 1")

    xy = np.mgrid[mini[0]:maxi[0]:0.001, mini[1]:maxi[1]:0.001].reshape(2,-1)

    class1x,class1y = [],[]
    class2x,class2y = [],[]
    class3x,class3y = [],[]

    X_data = np.transpose(xy)
    X_data_std = get_standardization(X_data, mean, std, standard)
    pred = np.squeeze(get_class(get_sigmoid(np.matmul(X_data_std,w))))
    plt.scatter(X_data[pred==0,0], X_data[pred==0,1], color='orangered')
    plt.scatter(X_data[pred==1,0], X_data[pred==1,1], color='lawngreen')
    
    plt.xlabel("x1")
    plt.ylabel("y1")
    plt.title("Decision boundary for Dataset "+dataset)

    plt.savefig("results/logreg/kernel/linear"+word+"/ds"+dataset+"boundary")