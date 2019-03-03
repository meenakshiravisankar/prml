import numpy as np 
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix
import confusion_matrix as cf_mat
import matplotlib.pyplot as plt

np.random.seed(seed=42)

def getPrior(y) :
    unique, counts = np.unique(y, return_counts=True)
    return unique, np.array(counts/sum(counts)).reshape(1,-1)

# According to MLE, estimate mean of distribution using sample mean
def getMLE(X,y) :
    unique = np.unique(y, return_counts=False)
    means = []
    for class_val in unique :
        means.append(np.mean(X[np.where(y==class_val)],axis=0))
    return means


def getRisk(lossfunction, classConditional, prior) :
    return np.transpose(np.matmul(lossfunction,np.transpose(np.multiply(classConditional,prior))))

def getConditional(X, mu, sigma, mode):
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

# compute for test set
def getModel(X, y, means, cov, lossfunction, prior, mode) :
    classConditional = getConditional(X, means, cov, mode)
    risk = getRisk(lossfunction, classConditional, prior)
    prediction = np.argmin(risk, axis=1)
    accuracies = np.sum(prediction == y)/y.shape[0]*100
    return prediction, accuracies

# read dataset
data = np.loadtxt("../Datasets_PRML_A1/Dataset_1_Team_39.csv", delimiter=',', dtype=None)

# shuffling
np.random.shuffle(data)

# splitting into train, test, validation - 80, 10, 10 and converting to numpy array
train_size = int(0.8*data.shape[0])
val_size = int(0.1*data.shape[0])
test_size = int(0.1*data.shape[0])

X_train = data[:train_size,:2]
y_train = data[:train_size,-1]

X_val = data[train_size:train_size+val_size,:2]
y_val = data[train_size:train_size+val_size,-1]

X_test = data[train_size+val_size:train_size+val_size+test_size,:2]
y_test = data[train_size+val_size:train_size+val_size+test_size,-1]

print("Size of train, validation and test sets",X_train.shape,X_val.shape,X_test.shape)
classes, prior = getPrior(y_train)
print("Number of classes", len(classes))
means = np.array(getMLE(X_train, y_train)) 
lossfunction = np.array([[0,1,2],[1,0,1],[2,1,0]])

print("\n")

accuracies = []

print("Model 1 - Naive Bayes and covariance is identity")
prediction, accuracy =  getModel(X_train, y_train, means, np.eye(2), lossfunction, prior, "naive")
print("Train accuracy {:.2f}".format(accuracy))
accuracies.append(accuracy)
prediction, accuracy =  getModel(X_val, y_val, means, np.eye(2), lossfunction, prior, "naive")
print("Validation accuracy {:.2f}".format(accuracy))
prediction, accuracy =  getModel(X_test, y_test, means, np.eye(2), lossfunction, prior, "naive")
print("Test accuracy {:.2f}".format(accuracy))
getConfusion(y_test,prediction, "Model 1")

print("\n")


print("Model 2 - Naive Bayes and covariance is same")
cov_rand = np.random.rand(1)*np.eye(2)
prediction, accuracy =  getModel(X_train, y_train, means, cov_rand, lossfunction, prior, "naive")
print("Train accuracy {:.2f}".format(accuracy))
accuracies.append(accuracy)
prediction, accuracy =  getModel(X_val, y_val, means, cov_rand, lossfunction, prior, "naive")
print("Validation accuracy {:.2f}".format(accuracy))
prediction, accuracy =  getModel(X_test, y_test, means, cov_rand, lossfunction, prior, "naive")
print("Test accuracy {:.2f}".format(accuracy))
getConfusion(y_test,prediction, "Model 2")

print("\n")


print("Model 3 - Naive Bayes and covariance different")
cov_rand = np.random.rand(2,2)
prediction, accuracy =  getModel(X_train, y_train, means, cov_rand, lossfunction, prior, "naive")
print("Train accuracy {:.2f}".format(accuracy))
accuracies.append(accuracy)
prediction, accuracy =  getModel(X_val, y_val, means, cov_rand, lossfunction, prior, "naive")
print("Validation accuracy {:.2f}".format(accuracy))
prediction, accuracy =  getModel(X_test, y_test, means, cov_rand, lossfunction, prior, "naive")
print("Test accuracy {:.2f}".format(accuracy))
getConfusion(y_test,prediction, "Model 3")

print("\n")


print("Model 4 -  Bayes and covariance is same")
cov_rand = np.random.rand(1)*np.eye(2)
prediction, accuracy =  getModel(X_train, y_train, means, cov_rand, lossfunction, prior, "bayes")
print("Train accuracy {:.2f}".format(accuracy))
accuracies.append(accuracy)
prediction, accuracy =  getModel(X_val, y_val, means, cov_rand, lossfunction, prior, "bayes")
print("Validation accuracy {:.2f}".format(accuracy))
prediction, accuracy =  getModel(X_test, y_test, means, cov_rand, lossfunction, prior, "bayes")
print("Test accuracy {:.2f}".format(accuracy))
getConfusion(y_test,prediction, "Model 4")

print("\n")

print("Model 5 - Bayes and covariance is identity")
cov_rand = np.random.rand(2,2)
prediction, accuracy =  getModel(X_train, y_train, means, cov_rand, lossfunction, prior, "bayes")
print("Train accuracy {:.2f}".format(accuracy))
accuracies.append(accuracy)
prediction, accuracy =  getModel(X_val, y_val, means, cov_rand, lossfunction, prior, "bayes")
print("Validation accuracy {:.2f}".format(accuracy))
prediction, accuracy =  getModel(X_test, y_test, means, cov_rand, lossfunction, prior, "bayes")
print("Test accuracy {:.2f}".format(accuracy))
getConfusion(y_test,prediction, "Model 5")

print("\n")

np.savetxt('results/accuracy_of_models',accuracies,fmt='%.2f')

# =======
# import numpy as np
# import matplotlib.pyplot as plt
# import functions as f

# np.random.seed(seed=42)

# # read dataset
# data = np.loadtxt("../Datasets_PRML_A1/Dataset_1_Team_39.csv", delimiter=',', dtype=None)

# # shuffling
# np.random.shuffle(data)

# # splitting into train, test, validation - 80, 10, 10 and converting to numpy array
# train_size = int(0.8*data.shape[0])
# val_size = int(0.1*data.shape[0])
# test_size = int(0.1*data.shape[0])

# X_train = data[:train_size,:2]
# y_train = data[:train_size,-1]

# X_val = data[train_size:train_size+val_size,:2]
# y_val = data[train_size:train_size+val_size,-1]

# X_test = data[train_size+val_size:train_size+val_size+test_size,:2]
# y_test = data[train_size+val_size:train_size+val_size+test_size,-1]

# print()
# print("Size of train, validation and test sets",X_train.shape,X_val.shape,X_test.shape)
# print()

# classes, prior = f.getPrior(y_train)
# print("Number of classes", len(classes))
# print()
# means = np.array(f.getMLE(X_train, y_train))
# lossfunction = np.array([[0,1,2],[1,0,1],[2,1,0]])

# accuracies = []

# print("Model 1 - Naive Bayes and covariance is identity")
# prediction, accuracy =  f.getModel(X_train, y_train, means, np.eye(2), lossfunction, prior, "naive")
# print("Train accuracy {:.2f}".format(accuracy))
# accuracies.append(accuracy)
# prediction, accuracy =  f.getModel(X_val, y_val, means, np.eye(2), lossfunction, prior, "naive")
# print("Validation accuracy {:.2f}".format(accuracy))
# prediction, accuracy =  f.getModel(X_test, y_test, means, np.eye(2), lossfunction, prior, "naive")
# print("Test accuracy {:.2f}".format(accuracy))
# #f.getConfusion(y_test,prediction, "Model 1")

# print("\n")


# print("Model 2 - Naive Bayes and covariance is same")
# cov_rand = f.getCovMatrix(np.transpose(X_train))
# prediction, accuracy =  f.getModel(X_train, y_train, means, cov_rand, lossfunction, prior, "naive")
# print("Train accuracy {:.2f}".format(accuracy))
# accuracies.append(accuracy)
# prediction, accuracy =  f.getModel(X_val, y_val, means, cov_rand, lossfunction, prior, "naive")
# print("Validation accuracy {:.2f}".format(accuracy))
# prediction, accuracy =  f.getModel(X_test, y_test, means, cov_rand, lossfunction, prior, "naive")
# print("Test accuracy {:.2f}".format(accuracy))
# #f.getConfusion(y_test,prediction, "Model 2")

# print("\n")


# print("Model 3 - Naive Bayes and covariance different")
# cov_rand = f.getCovMatrix(np.transpose(X_train)) # change this
# prediction, accuracy =  f.getModel(X_train, y_train, means, cov_rand, lossfunction, prior, "naive")
# print("Train accuracy {:.2f}".format(accuracy))
# accuracies.append(accuracy)
# prediction, accuracy =  f.getModel(X_val, y_val, means, cov_rand, lossfunction, prior, "naive")
# print("Validation accuracy {:.2f}".format(accuracy))
# prediction, accuracy =  f.getModel(X_test, y_test, means, cov_rand, lossfunction, prior, "naive")
# print("Test accuracy {:.2f}".format(accuracy))
# #f.getConfusion(y_test,prediction, "Model 3")

# print("\n")


# print("Model 4 -  Bayes and covariance is same")
# cov_rand = f.getCovMatrix(np.transpose(X_train))
# prediction, accuracy =  f.getModel(X_train, y_train, means, cov_rand, lossfunction, prior, "bayes")
# print("Train accuracy {:.2f}".format(accuracy))
# accuracies.append(accuracy)
# prediction, accuracy =  f.getModel(X_val, y_val, means, cov_rand, lossfunction, prior, "bayes")
# print("Validation accuracy {:.2f}".format(accuracy))
# prediction, accuracy =  f.getModel(X_test, y_test, means, cov_rand, lossfunction, prior, "bayes")
# print("Test accuracy {:.2f}".format(accuracy))
# #f.getConfusion(y_test,prediction, "Model 4")

# print("\n")

# print("Model 5 - Bayes and covariance is identity")
# print("Model 5 - Bayes and covariance different")
# cov_rand = f.getCovMatrix(np.transpose(X_train))
# prediction1, accuracy =  f.getModel(X_train, y_train, means, cov_rand, lossfunction, prior, "bayes")
# print("Train accuracy {:.2f}".format(accuracy))
# accuracies.append(accuracy)
# prediction, accuracy =  f.getModel(X_val, y_val, means, cov_rand, lossfunction, prior, "bayes")
# print("Validation accuracy {:.2f}".format(accuracy))
# prediction, accuracy =  f.getModel(X_test, y_test, means, cov_rand, lossfunction, prior, "bayes")
# print("Test accuracy {:.2f}".format(accuracy))
# #f.getConfusion(y_test,prediction, "Model 5")

# print("\n")

# np.savetxt('results/accuracy_of_models',accuracies,fmt='%.2f')

# l = prediction1.size
# z = prediction1*np.eye(l)
# z = np.array(z)
# z = z.reshape(l, l)
# plt.contour(X_train[:1, :], X_train[:-1, :], z, 50) # check this...
# # >>>>>>> f80f61576445ab19ae19ed91a7c38383d9b7fb47
