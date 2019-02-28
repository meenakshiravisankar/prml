import numpy as np
import matplotlib.pyplot as plt
import functions as f

np.random.seed(seed=42)

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
classes, prior = f.getPrior(y_train)
print("Number of classes", len(classes))
means = np.array(f.getMLE(X_train, y_train))
lossfunction = np.array([[0,1,2],[1,0,1],[2,1,0]])

print("\n")

accuracies = []

print("Model 1 - Naive Bayes and covariance is identity")
prediction, accuracy =  f.getModel(X_train, y_train, means, np.eye(2), lossfunction, prior, "naive")
print("Train accuracy {:.2f}".format(accuracy))
accuracies.append(accuracy)
prediction, accuracy =  f.getModel(X_val, y_val, means, np.eye(2), lossfunction, prior, "naive")
print("Validation accuracy {:.2f}".format(accuracy))
prediction, accuracy =  f.getModel(X_test, y_test, means, np.eye(2), lossfunction, prior, "naive")
print("Test accuracy {:.2f}".format(accuracy))
f.getConfusion(y_test,prediction, "Model 1")

print("\n")


print("Model 2 - Naive Bayes and covariance is same")
cov_rand = np.random.rand(1)*np.eye(2)
prediction, accuracy =  f.getModel(X_train, y_train, means, cov_rand, lossfunction, prior, "naive")
print("Train accuracy {:.2f}".format(accuracy))
accuracies.append(accuracy)
prediction, accuracy =  f.getModel(X_val, y_val, means, cov_rand, lossfunction, prior, "naive")
print("Validation accuracy {:.2f}".format(accuracy))
prediction, accuracy =  f.getModel(X_test, y_test, means, cov_rand, lossfunction, prior, "naive")
print("Test accuracy {:.2f}".format(accuracy))
f.getConfusion(y_test,prediction, "Model 2")

print("\n")


print("Model 3 - Naive Bayes and covariance different")
cov_rand = np.random.rand(2,2)
prediction, accuracy =  f.getModel(X_train, y_train, means, cov_rand, lossfunction, prior, "naive")
print("Train accuracy {:.2f}".format(accuracy))
accuracies.append(accuracy)
prediction, accuracy =  f.getModel(X_val, y_val, means, cov_rand, lossfunction, prior, "naive")
print("Validation accuracy {:.2f}".format(accuracy))
prediction, accuracy =  f.getModel(X_test, y_test, means, cov_rand, lossfunction, prior, "naive")
print("Test accuracy {:.2f}".format(accuracy))
f.getConfusion(y_test,prediction, "Model 3")

print("\n")


print("Model 4 -  Bayes and covariance is same")
cov_rand = np.random.rand(1)*np.eye(2)
prediction, accuracy =  f.getModel(X_train, y_train, means, cov_rand, lossfunction, prior, "bayes")
print("Train accuracy {:.2f}".format(accuracy))
accuracies.append(accuracy)
prediction, accuracy =  f.getModel(X_val, y_val, means, cov_rand, lossfunction, prior, "bayes")
print("Validation accuracy {:.2f}".format(accuracy))
prediction, accuracy =  f.getModel(X_test, y_test, means, cov_rand, lossfunction, prior, "bayes")
print("Test accuracy {:.2f}".format(accuracy))
f.getConfusion(y_test,prediction, "Model 4")

print("\n")

print("Model 5 - Bayes and covariance different")
cov_rand = np.random.rand(2,2)
prediction1, accuracy =  f.getModel(X_train, y_train, means, cov_rand, lossfunction, prior, "bayes")
print("Train accuracy {:.2f}".format(accuracy))
accuracies.append(accuracy)
prediction, accuracy =  f.getModel(X_val, y_val, means, cov_rand, lossfunction, prior, "bayes")
print("Validation accuracy {:.2f}".format(accuracy))
prediction, accuracy =  f.getModel(X_test, y_test, means, cov_rand, lossfunction, prior, "bayes")
print("Test accuracy {:.2f}".format(accuracy))
f.getConfusion(y_test,prediction, "Model 5")

print("\n")

np.savetxt('results/accuracy_of_models',accuracies,fmt='%.2f')

l = prediction1.size
z = prediction1*np.eye(l)
z = np.array(z)
z = z.reshape(l, l)
plt.contour(X_train[:1, :], X_train[:-1, :], z, 50) # check this...
