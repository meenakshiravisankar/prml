import numpy as np
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix
import confusion_matrix as cf_mat
import matplotlib.pyplot as plt

np.random.seed(seed=42)

# Compute class prior
def getPrior(y) :
    unique, counts = np.unique(y, return_counts=True)
    # print(unique)
    return unique, np.array(counts/sum(counts)).reshape(1,-1)

# According to MLE, estimate mean of distribution using sample mean
def getMLE(X,y) :
    unique = np.unique(y, return_counts=False)
    means = []
    for class_val in unique :
        means.append(np.mean(X[np.where(y==class_val)],axis=0))
    return means

# Compute the risk for each class
def getRisk(lossfunction, classConditional, prior) :
    return np.transpose(np.matmul(lossfunction,np.transpose(np.multiply(classConditional,prior))))

# Compute class conditional density where covariance is same for all classes
def getConditionalSameCov(X, mu, sigma, mode):
    prob = []
    for class_val in range(mu.shape[0]) :
        if mode == "naive" :
            value = 1
            for feature in range(mu.shape[1]) :
                value *= (multivariate_normal.pdf(X[:,feature],mean=mu[class_val][feature],cov=sigma[feature][feature]))
        else :
            value =  multivariate_normal.pdf(X,mean=mu[class_val],cov=sigma)
        prob.append(value)
    return np.transpose(np.array(prob))


# Compute class conditional density where covariance is different for all classes
def getConditionalDiffCov(X, mu, sigma, mode):
    prob = []
    for class_val in range(mu.shape[0]) :
        if mode == "naive" :
            value = 1
            for feature in range(mu.shape[1]) :
                value *= (multivariate_normal.pdf(X[:,feature],mean=mu[class_val][feature],cov=sigma[class_val][feature][feature]))
        else :
            value =  multivariate_normal.pdf(X,mean=mu[class_val],cov=sigma[class_val])
        prob.append(value)
    return np.transpose(np.array(prob))

# Compute confusion matrix
def getConfusion(y_test, prediction, name) :
    # confusion matrix for test
    cnf_matrix = confusion_matrix(y_test, prediction)
    class_names = np.unique(prediction, return_counts=False)
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plt.figure()
    cf_mat.plot_confusion_matrix(cnf_matrix, classes=class_names,title=name)
    plt.savefig("results/"+name)
    # plt.show()
    return

# compute posterior
def getModel(X, y, means, cov, lossfunction, prior, mode, covmode) :
    if covmode == "same":
        classConditional = getConditionalSameCov(X, means, cov, mode)
    else:
        classConditional = getConditionalDiffCov(X, means, cov, mode)
    risk = getRisk(lossfunction, classConditional, prior)
    prediction = np.argmin(risk, axis=1)
    accuracies = np.sum(prediction == y)/y.shape[0]*100
    return prediction, accuracies

def mean(X):
    n = len(X)
    mean_x = 0.0
    for i in range(n):
        mean_x += X[i]
    mean_x = mean_x/n
    return mean_x

# Compute covariance of single feature of particular class
def getCovariance(X1, X2):
    Z = []
    n = len(X1)
    for i in range(n):
        Z.append(X1[i]*X2[i])
    e_x1x2 = mean(Z)
    e_x1 = mean(X1)
    e_x2 = mean(X2)
    cov = e_x1x2 - (e_x1*e_x2)
    return cov

# Compute covariance for single class
def getCovMatrix(X):
    n = len(X)
    cov_mat = []
    for i in range(n):
        for j in range(n):
            cov_mat.append(getCovariance(X[i], X[j]))
    return np.reshape(cov_mat, (n, n))

# Compute covariance matrix for all classes
def getCompleteCovMatrix(X, y):
    unique = np.unique(y, return_counts=False)
    covs = []
    for class_val in unique :
        covs.append(getCovMatrix(np.transpose(X[np.where(y==class_val)])))
    return covs

def getContour(mini, maxi, mu, cov, color, fig) :
    #Create grid and multivariate normal
    x = np.linspace(mini[0],maxi[0],500)
    y = np.linspace(mini[1],maxi[1],500)
    X, Y = np.meshgrid(x,y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y
    rv = multivariate_normal(mu, cov)
    return X,Y,rv.pdf(pos)

def getSurfacePlot(mini, maxi, mu, cov, color, fig) :

    #Create grid and multivariate normal
    x = np.linspace(mini[0],maxi[0],500)
    y = np.linspace(mini[1],maxi[1],500)
    X, Y = np.meshgrid(x,y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y
    rv = multivariate_normal(mu, cov)

    #Make a 3D plot
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, rv.pdf(pos),linewidth=0,cmap=color)
    ax.ticklabel_format(style='sci', axis='z',scilimits=(0,1))

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel(r'$f(x|class)$')
    return X,Y,rv.pdf(pos)

# read datasets
data1 = np.loadtxt("../Datasets_PRML_A1/Dataset_1_Team_39.csv", delimiter=',', dtype=None)
data2 = np.loadtxt("../Datasets_PRML_A1/Dataset_2_Team_39.csv", delimiter=',', dtype=None)


best_model_1 = [0]
best_model_2 = [0]

# shuffling
np.random.shuffle(data1)

# splitting into train, test, validation - 70, 15, 15
train_size = int(0.7*data1.shape[0])
val_size = int(0.15*data1.shape[0])
test_size = int(0.15*data1.shape[0])

X_train = data1[:train_size,:2]
y_train = data1[:train_size,-1]

X_val = data1[train_size:train_size+val_size,:2]
y_val = data1[train_size:train_size+val_size,-1]

X_test = data1[train_size+val_size:train_size+val_size+test_size,:2]
y_test = data1[train_size+val_size:train_size+val_size+test_size,-1]


mini = np.min(X_train, axis=0)
maxi = np.max(X_train, axis=0)




print("Dataset 1")
print("Size of train, validation and test sets",X_train.shape,X_val.shape,X_test.shape)
classes, prior = getPrior(y_train)
print("Number of classes", len(classes))
means = np.array(getMLE(X_train, y_train))
lossfunction = np.array([[0,1,2],[1,0,1],[2,1,0]])

print("\n")

train_accuracies = []
val_accuracies = []

print("Model 1 - Naive Bayes and covariance is identity")
prediction, accuracy =  getModel(X_train, y_train, means, np.eye(2), lossfunction, prior, "naive", "same")
train_accuracies.append(accuracy)
print("Train accuracy {:.2f}".format(accuracy))
prediction, accuracy =  getModel(X_val, y_val, means, np.eye(2), lossfunction, prior, "naive", "same")
print("Validation accuracy {:.2f}".format(accuracy))
val_accuracies.append(accuracy)
if best_model_1[0] < accuracy :
    best_model_1 = [accuracy, "naive", "same covariance - identity"]
prediction, accuracy =  getModel(X_test, y_test, means, np.eye(2), lossfunction, prior, "naive", "same")
print("Test accuracy {:.2f}".format(accuracy))
# getConfusion(y_test,prediction, "Model 1 Dataset 1")

print("\n")

print("Model 2 - Naive Bayes and covariance is same")
cov_rand = getCovMatrix(np.transpose(X_train))
prediction, accuracy =  getModel(X_train, y_train, means, cov_rand, lossfunction, prior, "naive", "same")
train_accuracies.append(accuracy)
print("Train accuracy {:.2f}".format(accuracy))
prediction, accuracy =  getModel(X_val, y_val, means, cov_rand, lossfunction, prior, "naive", "same")
print("Validation accuracy {:.2f}".format(accuracy))
val_accuracies.append(accuracy)
if best_model_1[0] < accuracy :
    best_model_1 = [accuracy, "naive", "same covariance"]
prediction, accuracy =  getModel(X_test, y_test, means, cov_rand, lossfunction, prior, "naive", "same")
print("Test accuracy {:.2f}".format(accuracy))

# fig = plt.figure()
# ax = fig.gca(projection='3d')

# ax.scatter(X_train[y_train==0][:,0],X_train[y_train==0][:,1],c='k')
# ax.scatter(X_train[y_train==1][:,0],X_train[y_train==1][:,1],c='b')
# ax.scatter(X_train[y_train==2][:,0],X_train[y_train==2][:,1],c='g')


# x,y,z = getSurfacePlot(mini,maxi,means[0,:],cov_rand,"Reds", fig)
# x,y,z = getSurfacePlot(mini,maxi,means[1,:],cov_rand,"Blues", fig)
# x,y,z = getSurfacePlot(mini,maxi,means[2,:],cov_rand,"Greens", fig)
# plt.savefig("results/q11density.png")
# plt.show()

# fig = plt.figure()
# x,y,z = getContour(mini,maxi,means[0,:],cov_rand,"Reds", fig)
# plt.contour(x,y,z,colors=['red'])
# x,y,z = getContour(mini,maxi,means[1,:],cov_rand,"Blues", fig)
# plt.contour(x,y,z,colors=['blue'])
# x,y,z = getContour(mini,maxi,means[2,:],cov_rand,"Greens", fig)
# plt.contour(x,y,z,colors=['green'])
# plt.show()
plt.subplot(121)
getConfusion(y_test,prediction, "Model 2 Dataset 1")

print("\n")

print("Model 3 - Naive Bayes and covariance different")
cov_rand = getCompleteCovMatrix(X_train, y_train)
prediction, accuracy =  getModel(X_train, y_train, means, cov_rand, lossfunction, prior, "naive", "diff")
train_accuracies.append(accuracy)
print("Train accuracy {:.2f}".format(accuracy))
prediction, accuracy =  getModel(X_val, y_val, means, cov_rand, lossfunction, prior, "naive", "diff")
print("Validation accuracy {:.2f}".format(accuracy))
val_accuracies.append(accuracy)
if best_model_1[0] < accuracy :
    best_model_1 = [accuracy, "naive", "different covariance"]
prediction, accuracy =  getModel(X_test, y_test, means, cov_rand, lossfunction, prior, "naive", "diff")
print("Test accuracy {:.2f}".format(accuracy))
# getConfusion(y_test,prediction, "Model 3 Dataset 1")

print("\n")

print("Model 4 -  Bayes and covariance is same")
cov_rand = getCovMatrix(np.transpose(X_train))
prediction, accuracy =  getModel(X_train, y_train, means, cov_rand, lossfunction, prior, "bayes", "same")
train_accuracies.append(accuracy)
print("Train accuracy {:.2f}".format(accuracy))
prediction, accuracy =  getModel(X_val, y_val, means, cov_rand, lossfunction, prior, "bayes", "same")
print("Validation accuracy {:.2f}".format(accuracy))
val_accuracies.append(accuracy)
if best_model_1[0] < accuracy :
    best_model_1 = [accuracy, "bayes", "same covariance"]
prediction, accuracy =  getModel(X_test, y_test, means, cov_rand, lossfunction, prior, "bayes", "same")
print("Test accuracy {:.2f}".format(accuracy))
# getConfusion(y_test,prediction, "Model 4 Dataset 1")

print("\n")

print("Model 5 - Bayes and covariance is different")
cov_rand = getCompleteCovMatrix(X_train, y_train)
prediction, accuracy =  getModel(X_train, y_train, means, cov_rand, lossfunction, prior, "bayes", "diff")
train_accuracies.append(accuracy)
print("Train accuracy {:.2f}".format(accuracy))
prediction, accuracy =  getModel(X_val, y_val, means, cov_rand, lossfunction, prior, "bayes", "diff")
print("Validation accuracy {:.2f}".format(accuracy))
val_accuracies.append(accuracy)
if best_model_1[0] < accuracy :
    best_model_1 = [accuracy, "bayes", "different covariance"]
prediction, accuracy =  getModel(X_test, y_test, means, cov_rand, lossfunction, prior, "bayes", "diff")
print("Test accuracy {:.2f}".format(accuracy))
# getConfusion(y_test,prediction, "Model 5 Dataset 1")

print("\n")

np.savetxt('results/accuracy_of_training_dataset_1',train_accuracies,fmt='%.2f')
np.savetxt('results/accuracy_of_validation_dataset_1',val_accuracies,fmt='%.2f')

# shuffling
np.random.shuffle(data2)

train_accuracies = []
val_accuracies = []

# splitting into train, test, validation - 80, 10, 10 and converting to numpy array
train_size = int(0.7*data2.shape[0])
val_size = int(0.15*data2.shape[0])
test_size = int(0.15*data2.shape[0])

X_train = data2[:train_size,:2]
y_train = data2[:train_size,-1]

X_val = data2[train_size:train_size+val_size,:2]
y_val = data2[train_size:train_size+val_size,-1]

X_test = data2[train_size+val_size:train_size+val_size+test_size,:2]
y_test = data2[train_size+val_size:train_size+val_size+test_size,-1]

print("Dataset 2")
print("Size of train, validation and test sets",X_train.shape,X_val.shape,X_test.shape)
classes, prior = getPrior(y_train)
print("Number of classes", len(classes))
means = np.array(getMLE(X_train, y_train))
lossfunction = np.array([[0,1,2],[1,0,1],[2,1,0]])

print("\n")

accuracies = []

print("Model 1 - Naive Bayes and covariance is identity")
prediction, accuracy =  getModel(X_train, y_train, means, np.eye(2), lossfunction, prior, "naive", "same")
train_accuracies.append(accuracy)
print("Train accuracy {:.2f}".format(accuracy))
prediction, accuracy =  getModel(X_val, y_val, means, np.eye(2), lossfunction, prior, "naive", "same")
print("Validation accuracy {:.2f}".format(accuracy))
val_accuracies.append(accuracy)
if best_model_2[0] < accuracy :
    best_model_2 = [accuracy, "naive", "same covariance - identity"]
prediction, accuracy =  getModel(X_test, y_test, means, np.eye(2), lossfunction, prior, "naive", "same")
print("Test accuracy {:.2f}".format(accuracy))
# getConfusion(y_test,prediction, "Model 1 Dataset 2")

print("\n")


print("Model 2 - Naive Bayes and covariance is same")
cov_rand = getCovMatrix(np.transpose(X_train))
prediction, accuracy =  getModel(X_train, y_train, means, cov_rand, lossfunction, prior, "naive", "same")
train_accuracies.append(accuracy)
print("Train accuracy {:.2f}".format(accuracy))
prediction, accuracy =  getModel(X_val, y_val, means, cov_rand, lossfunction, prior, "naive", "same")
print("Validation accuracy {:.2f}".format(accuracy))
val_accuracies.append(accuracy)
if best_model_2[0] < accuracy :
    best_model_2 = [accuracy, "naive", "same covariance"]
prediction, accuracy =  getModel(X_test, y_test, means, cov_rand, lossfunction, prior, "naive", "same")
print("Test accuracy {:.2f}".format(accuracy))
# getConfusion(y_test,prediction, "Model 2 Dataset 2")

print("\n")


print("Model 3 - Naive Bayes and covariance different")
cov_rand = getCompleteCovMatrix(X_train, y_train)
prediction, accuracy =  getModel(X_train, y_train, means, cov_rand, lossfunction, prior, "naive", "diff")
train_accuracies.append(accuracy)
print("Train accuracy {:.2f}".format(accuracy))
prediction, accuracy =  getModel(X_val, y_val, means, cov_rand, lossfunction, prior, "naive", "diff")
print("Validation accuracy {:.2f}".format(accuracy))
val_accuracies.append(accuracy)
if best_model_2[0] < accuracy :
    best_model_2 = [accuracy, "naive", "different covariance"]
prediction, accuracy =  getModel(X_test, y_test, means, cov_rand, lossfunction, prior, "naive", "diff")
print("Test accuracy {:.2f}".format(accuracy))
# getConfusion(y_test,prediction, "Model 3 Dataset 2")

print("\n")


print("Model 4 -  Bayes and covariance is same")
cov_rand = getCovMatrix(np.transpose(X_train))
prediction, accuracy =  getModel(X_train, y_train, means, cov_rand, lossfunction, prior, "bayes", "same")
train_accuracies.append(accuracy)
print("Train accuracy {:.2f}".format(accuracy))
prediction, accuracy =  getModel(X_val, y_val, means, cov_rand, lossfunction, prior, "bayes", "same")
print("Validation accuracy {:.2f}".format(accuracy))
val_accuracies.append(accuracy)
if best_model_2[0] < accuracy :
    best_model_2 = [accuracy, "bayes", "same covariance"]
prediction, accuracy =  getModel(X_test, y_test, means, cov_rand, lossfunction, prior, "bayes", "same")
print("Test accuracy {:.2f}".format(accuracy))
# getConfusion(y_test,prediction, "Model 4 Dataset 2")

print("\n")

print("Model 5 - Bayes and covariance is different")
cov_rand = getCompleteCovMatrix(X_train, y_train)
prediction, accuracy =  getModel(X_train, y_train, means, cov_rand, lossfunction, prior, "bayes", "diff")
train_accuracies.append(accuracy)
print("Train accuracy {:.2f}".format(accuracy))
prediction, accuracy =  getModel(X_val, y_val, means, cov_rand, lossfunction, prior, "bayes", "diff")
print("Validation accuracy {:.2f}".format(accuracy))
val_accuracies.append(accuracy)
if best_model_2[0] < accuracy :
    best_model_2 = [accuracy, "bayes", "different covariance"]
prediction, accuracy =  getModel(X_test, y_test, means, cov_rand, lossfunction, prior, "bayes", "diff")
print("Test accuracy {:.2f}".format(accuracy))
# plt.subplot(122)
# getConfusion(y_test,prediction, "Model 5 Dataset 2")

fig = plt.figure()
cov_rand = np.array(cov_rand)
getSurfacePlot(mini,maxi,means[0,:],cov_rand[0,:,:],"Reds", fig)
getSurfacePlot(mini,maxi,means[1,:],cov_rand[1,:,:],"Blues", fig)
getSurfacePlot(mini,maxi,means[2,:],cov_rand[2,:,:],"Greens", fig)
plt.savefig("results/q12density.png")
plt.show()
print("\n")

print("The best model for dataset 1 has validation accuracy ",best_model_1)
print("The best model for dataset 2 has validation accuracy ",best_model_2)
np.savetxt('results/accuracy_of_training_dataset_2',train_accuracies,fmt='%.2f')
np.savetxt('results/accuracy_of_validation_dataset_2',val_accuracies,fmt='%.2f')

# plt.show()