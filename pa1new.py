import numpy as np
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix
import confusion_matrix as cf_mat
import matplotlib.pyplot as plt
import functions as f

np.random.seed(seed=42)

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
classes, prior = f.getPrior(y_train)
print("Number of classes", len(classes))
means = np.array(f.getMLE(X_train, y_train))
lossfunction = np.array([[0,1,2],[1,0,1],[2,1,0]])

print("\n")

train_accuracies = []
val_accuracies = []

print("Model 1 - Naive Bayes and covariance is identity")
prediction, accuracy =  f.getModel(X_train, y_train, means, np.eye(2), lossfunction, prior, "naive", "same")
train_accuracies.append(accuracy)
print("Train accuracy {:.2f}".format(accuracy))
prediction, accuracy =  f.getModel(X_val, y_val, means, np.eye(2), lossfunction, prior, "naive", "same")
print("Validation accuracy {:.2f}".format(accuracy))
val_accuracies.append(accuracy)
if best_model_1[0] < accuracy :
    best_model_1 = [accuracy, "naive", "same covariance - identity"]
prediction, accuracy =  f.getModel(X_test, y_test, means, np.eye(2), lossfunction, prior, "naive", "same")
print("Test accuracy {:.2f}".format(accuracy))
# getConfusion(y_test,prediction, "Model 1 Dataset 1")

print("\n")

print("Model 2 - Naive Bayes and covariance is same")
cov_rand = f.getCovMatrix(np.transpose(X_train))
prediction, accuracy =  f.getModel(X_train, y_train, means, cov_rand, lossfunction, prior, "naive", "same")
train_accuracies.append(accuracy)
print("Train accuracy {:.2f}".format(accuracy))
prediction, accuracy =  f.getModel(X_val, y_val, means, cov_rand, lossfunction, prior, "naive", "same")
print("Validation accuracy {:.2f}".format(accuracy))
val_accuracies.append(accuracy)
if best_model_1[0] < accuracy :
    best_model_1 = [accuracy, "naive", "same covariance"]
prediction, accuracy =  f.getModel(X_test, y_test, means, cov_rand, lossfunction, prior, "naive", "same")
print("Test accuracy {:.2f}".format(accuracy))

fig = plt.figure()
# ax = fig.gca(projection='3d')

# Train scatter plot
plt.plot(X_train[y_train==0][:,0],X_train[y_train==0][:,1],'.',c='r',label="class 0")
plt.plot(X_train[y_train==1][:,0],X_train[y_train==1][:,1],'.',c='b',label="class 1")
plt.plot(X_train[y_train==2][:,0],X_train[y_train==2][:,1],'.',c='g',label="class 2")
# plt.show()

# Surface plot for class conditional density
# x,y,z = getSurfacePlot(mini,maxi,means[0,:],cov_rand,"Reds", fig)
# x,y,z = getSurfacePlot(mini,maxi,means[1,:],cov_rand,"Blues", fig)
# x,y,z = getSurfacePlot(mini,maxi,means[2,:],cov_rand,"Greens", fig)
# plt.savefig("results/q11density.png")
# plt.show()

# Contour plots
# plt.figure()
# x,y,z = getContour(mini,maxi,means[0,:],cov_rand,"Reds", fig)
# plt.contour(x,y,z,colors=['red'])
# x,y,z = getContour(mini,maxi,means[1,:],cov_rand,"Blues", fig)
# plt.contour(x,y,z,colors=['blue'])
# x,y,z = getContour(mini,maxi,means[2,:],cov_rand,"Greens", fig)
# plt.contour(x,y,z,colors=['green'])

# # # Eigen values

# for i in classes :
#     i = int(i)
#     w, v = np.linalg.eig(cov_rand)
#     plt.quiver(means[i,0],means[i,1],v[0][0],v[1][0],scale=3)
#     plt.quiver(means[i,0],means[i,1],v[0][1],v[1][1],scale=8)
# plt.xlabel("x1")
# plt.ylabel("x2")
# plt.title("Best model on dataset 1 - Naive with same covariance")
# plt.legend(loc="lower right")
# plt.savefig("results/q11contour.png")
# plt.show()

# plt.figure()
# plt.quiver(1,1,0.5,0.5,scale=0.001)
# plt.show()
# plt.subplot(121)
# getConfusion(y_test,prediction, "Model 2 Dataset 1")

# print("\n")

# print("Model 3 - Naive Bayes and covariance different")
# cov_rand = f.getCompleteCovMatrix(X_train, y_train)
# prediction, accuracy =  f.getModel(X_train, y_train, means, cov_rand, lossfunction, prior, "naive", "diff")
# train_accuracies.append(accuracy)
# print("Train accuracy {:.2f}".format(accuracy))
# prediction, accuracy =  f.getModel(X_val, y_val, means, cov_rand, lossfunction, prior, "naive", "diff")
# print("Validation accuracy {:.2f}".format(accuracy))
# val_accuracies.append(accuracy)
# if best_model_1[0] < accuracy :
#     best_model_1 = [accuracy, "naive", "different covariance"]
# prediction, accuracy =  f.getModel(X_test, y_test, means, cov_rand, lossfunction, prior, "naive", "diff")
# print("Test accuracy {:.2f}".format(accuracy))
# # getConfusion(y_test,prediction, "Model 3 Dataset 1")

# print("\n")

# print("Model 4 -  Bayes and covariance is same")
# cov_rand = f.getCovMatrix(np.transpose(X_train))
# prediction, accuracy =  f.getModel(X_train, y_train, means, cov_rand, lossfunction, prior, "bayes", "same")
# train_accuracies.append(accuracy)
# print("Train accuracy {:.2f}".format(accuracy))
# prediction, accuracy =  f.getModel(X_val, y_val, means, cov_rand, lossfunction, prior, "bayes", "same")
# print("Validation accuracy {:.2f}".format(accuracy))
# val_accuracies.append(accuracy)
# if best_model_1[0] < accuracy :
#     best_model_1 = [accuracy, "bayes", "same covariance"]
# prediction, accuracy =  f.getModel(X_test, y_test, means, cov_rand, lossfunction, prior, "bayes", "same")
# print("Test accuracy {:.2f}".format(accuracy))
# # getConfusion(y_test,prediction, "Model 4 Dataset 1")

# print("\n")

# print("Model 5 - Bayes and covariance is different")
# cov_rand = f.getCompleteCovMatrix(X_train, y_train)
# prediction, accuracy =  f.getModel(X_train, y_train, means, cov_rand, lossfunction, prior, "bayes", "diff")
# train_accuracies.append(accuracy)
# print("Train accuracy {:.2f}".format(accuracy))
# prediction, accuracy =  f.getModel(X_val, y_val, means, cov_rand, lossfunction, prior, "bayes", "diff")
# print("Validation accuracy {:.2f}".format(accuracy))
# val_accuracies.append(accuracy)
# if best_model_1[0] < accuracy :
#     best_model_1 = [accuracy, "bayes", "different covariance"]
# prediction, accuracy =  f.getModel(X_test, y_test, means, cov_rand, lossfunction, prior, "bayes", "diff")
# print("Test accuracy {:.2f}".format(accuracy))
# # getConfusion(y_test,prediction, "Model 5 Dataset 1")

# print("\n")

# np.savetxt('results/accuracy_of_training_dataset_1',train_accuracies,fmt='%.2f')
# np.savetxt('results/accuracy_of_validation_dataset_1',val_accuracies,fmt='%.2f')

# # shuffling
# np.random.shuffle(data2)

# train_accuracies = []
# val_accuracies = []

# # splitting into train, test, validation - 80, 10, 10 and converting to numpy array
# train_size = int(0.7*data2.shape[0])
# val_size = int(0.15*data2.shape[0])
# test_size = int(0.15*data2.shape[0])

# X_train = data2[:train_size,:2]
# y_train = data2[:train_size,-1]

# X_val = data2[train_size:train_size+val_size,:2]
# y_val = data2[train_size:train_size+val_size,-1]

# X_test = data2[train_size+val_size:train_size+val_size+test_size,:2]
# y_test = data2[train_size+val_size:train_size+val_size+test_size,-1]

# print("Dataset 2")
# print("Size of train, validation and test sets",X_train.shape,X_val.shape,X_test.shape)
# classes, prior = f.getPrior(y_train)
# print("Number of classes", len(classes))
# means = np.array(f.getMLE(X_train, y_train))
# lossfunction = np.array([[0,1,2],[1,0,1],[2,1,0]])

# print("\n")

# accuracies = []

# print("Model 1 - Naive Bayes and covariance is identity")
# prediction, accuracy =  f.getModel(X_train, y_train, means, np.eye(2), lossfunction, prior, "naive", "same")
# train_accuracies.append(accuracy)
# print("Train accuracy {:.2f}".format(accuracy))
# prediction, accuracy =  f.getModel(X_val, y_val, means, np.eye(2), lossfunction, prior, "naive", "same")
# print("Validation accuracy {:.2f}".format(accuracy))
# val_accuracies.append(accuracy)
# if best_model_2[0] < accuracy :
#     best_model_2 = [accuracy, "naive", "same covariance - identity"]
# prediction, accuracy =  f.getModel(X_test, y_test, means, np.eye(2), lossfunction, prior, "naive", "same")
# print("Test accuracy {:.2f}".format(accuracy))
# # getConfusion(y_test,prediction, "Model 1 Dataset 2")

# print("\n")


# print("Model 2 - Naive Bayes and covariance is same")
# cov_rand = f.getCovMatrix(np.transpose(X_train))
# prediction, accuracy =  f.getModel(X_train, y_train, means, cov_rand, lossfunction, prior, "naive", "same")
# train_accuracies.append(accuracy)
# print("Train accuracy {:.2f}".format(accuracy))
# prediction, accuracy =  f.getModel(X_val, y_val, means, cov_rand, lossfunction, prior, "naive", "same")
# print("Validation accuracy {:.2f}".format(accuracy))
# val_accuracies.append(accuracy)
# if best_model_2[0] < accuracy :
#     best_model_2 = [accuracy, "naive", "same covariance"]
# prediction, accuracy =  f.getModel(X_test, y_test, means, cov_rand, lossfunction, prior, "naive", "same")
# print("Test accuracy {:.2f}".format(accuracy))
# # getConfusion(y_test,prediction, "Model 2 Dataset 2")

# print("\n")


# print("Model 3 - Naive Bayes and covariance different")
# cov_rand = f.getCompleteCovMatrix(X_train, y_train)
# prediction, accuracy =  f.getModel(X_train, y_train, means, cov_rand, lossfunction, prior, "naive", "diff")
# train_accuracies.append(accuracy)
# print("Train accuracy {:.2f}".format(accuracy))
# prediction, accuracy =  f.getModel(X_val, y_val, means, cov_rand, lossfunction, prior, "naive", "diff")
# print("Validation accuracy {:.2f}".format(accuracy))
# val_accuracies.append(accuracy)
# if best_model_2[0] < accuracy :
#     best_model_2 = [accuracy, "naive", "different covariance"]
# prediction, accuracy =  f.getModel(X_test, y_test, means, cov_rand, lossfunction, prior, "naive", "diff")
# print("Test accuracy {:.2f}".format(accuracy))
# # getConfusion(y_test,prediction, "Model 3 Dataset 2")

# print("\n")


# print("Model 4 -  Bayes and covariance is same")
# cov_rand = f.getCovMatrix(np.transpose(X_train))
# prediction, accuracy =  f.getModel(X_train, y_train, means, cov_rand, lossfunction, prior, "bayes", "same")
# train_accuracies.append(accuracy)
# print("Train accuracy {:.2f}".format(accuracy))
# prediction, accuracy =  f.getModel(X_val, y_val, means, cov_rand, lossfunction, prior, "bayes", "same")
# print("Validation accuracy {:.2f}".format(accuracy))
# val_accuracies.append(accuracy)
# if best_model_2[0] < accuracy :
#     best_model_2 = [accuracy, "bayes", "same covariance"]
# prediction, accuracy =  f.getModel(X_test, y_test, means, cov_rand, lossfunction, prior, "bayes", "same")
# print("Test accuracy {:.2f}".format(accuracy))
# # getConfusion(y_test,prediction, "Model 4 Dataset 2")

# print("\n")

print("Model 5 - Bayes and covariance is different")
cov_rand = f.getCompleteCovMatrix(X_train, y_train)
prediction, accuracy =  f.getModel(X_train, y_train, means, cov_rand, lossfunction, prior, "bayes", "diff")
train_accuracies.append(accuracy)
print("Train accuracy {:.2f}".format(accuracy))
prediction, accuracy =  f.getModel(X_val, y_val, means, cov_rand, lossfunction, prior, "bayes", "diff")
print("Validation accuracy {:.2f}".format(accuracy))
val_accuracies.append(accuracy)
if best_model_2[0] < accuracy :
    best_model_2 = [accuracy, "bayes", "different covariance"]
prediction, accuracy =  f.getModel(X_test, y_test, means, cov_rand, lossfunction, prior, "bayes", "diff")
print("Test accuracy {:.2f}".format(accuracy))
# # plt.subplot(122)
# # getConfusion(y_test,prediction, "Model 5 Dataset 2")

# fig = plt.figure()
cov_rand = np.array(cov_rand)
# # getSurfacePlot(mini,maxi,means[0,:],cov_rand[0,:,:],"Reds", fig)
# # getSurfacePlot(mini,maxi,means[1,:],cov_rand[1,:,:],"Blues", fig)
# # getSurfacePlot(mini,maxi,means[2,:],cov_rand[2,:,:],"Greens", fig)
# # plt.savefig("results/q12density.png")
# # plt.show()

x,y,z = getContour(mini,maxi,means[0,:],cov_rand[0,:,:],"Reds", fig)
plt.contour(x,y,z,colors=['red'])
x,y,z = getContour(mini,maxi,means[1,:],cov_rand[1,:,:],"Blues", fig)
plt.contour(x,y,z,colors=['blue'])
x,y,z = getContour(mini,maxi,means[2,:],cov_rand[2,:,:],"Greens", fig)
plt.contour(x,y,z,colors=['green'])

# # Eigen values
print(cov_rand)
for i in classes :
    i = int(i)
    w, v = np.linalg.eig(cov_rand[i,:,:])
    plt.quiver(means[i,0],means[i,1],v[0][0],v[1][0],scale=3)
    plt.quiver(means[i,0],means[i,1],v[0][1],v[1][1],scale=8)
    print(v)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Best model on dataset 2 - Bayes with different covariance")
plt.legend(loc="lower right")
plt.savefig("results/q12contour.png")
plt.show()

# print("\n")

# print("The best model for dataset 1 has validation accuracy ",best_model_1)
# print("The best model for dataset 2 has validation accuracy ",best_model_2)
# np.savetxt('results/accuracy_of_training_dataset_2',train_accuracies,fmt='%.2f')
# np.savetxt('results/accuracy_of_validation_dataset_2',val_accuracies,fmt='%.2f')

# # plt.show()
