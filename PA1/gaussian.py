import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from scipy import linalg
import warnings
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# warnings.filterwarnings("ignore")
def getPolyfit(X,w) :
    return np.matmul(X,w)

# compute maximum likelihood estimate
def getWeights(X,y,ridge) :
    return np.matmul(linalg.inv(ridge*np.eye(X.shape[1]) + np.matmul(np.transpose(X),X)), np.matmul(np.transpose(X),y))

# computes gaussian basis functions with mus as mean
def getGaussianBasis(X,mus,sigma) :
    diff = X[:,None] - mus
    return np.exp(-np.linalg.norm(diff,axis=2)**2 / (2*(sigma**2)))

# computes Erms
def getRMSE(ytrue,ypred,ridge,w) :
    return np.sqrt((np.sum(np.multiply(ytrue-ypred,ytrue-ypred)) + ridge*np.sum(np.multiply(w,w)))/ytrue.shape[0])

# performs kmeans clustering and returns mean of clusters
def kmeans(dataset, k):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(dataset)
    return kmeans.cluster_centers_

# performs linear regression with gaussian basis functions 
def getResults(X,y,mu,sigma,ridge) :
    X = getGaussianBasis(X,mu,sigma)
    w = getWeights(X, y, ridge)
    y_pred = getPolyfit(X,w)
    rmse = getRMSE(y,y_pred,ridge,w)
    return w, y_pred, rmse

# Seeding
np.random.seed(42)

data100 = np.loadtxt("../Datasets_PRML_A1/train100.txt", delimiter=' ', dtype=None)
data1000 = np.loadtxt("../Datasets_PRML_A1/train1000.txt", delimiter=' ', dtype=None)
data2000 = np.loadtxt("../Datasets_PRML_A1/train.txt", delimiter=' ', dtype=None)

data2 = np.loadtxt("../Datasets_PRML_A1/val.txt", delimiter=' ', dtype=None)
data3 = np.loadtxt("../Datasets_PRML_A1/test.txt", delimiter=' ', dtype=None)

np.random.shuffle(data100)

X_train = data100[:,:2]
y_train = data100[:,-1]
X_val = data2[:,:2]
y_val = data2[:,-1]
X_test = data3[:,:2]
y_test = data3[:,-1]

mini = np.min(X_train,axis=0)-1
maxi = np.max(X_train,axis=0)+1

# Settings 
clusters = [2,4,10,20,30,40,50,60]
ridges = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
sigmas = [0.1,1,2]
ridges[0] = 0

default = 0
emrs = 0
ridge_go = 0
go = 0

if default :
    clusters = [60]
    ridges = [0]
    sigmas = [2]
if emrs :
    clusters = [10,30,60,70,80]
    ridges = [0] # 0.001 and 0
    sigmas = [2]

train_rmses = []
val_rmses = []
test_rmses = []

for cluster in clusters :
    for ridge in ridges :
        for sigma in sigmas :
            mu = kmeans(X_train,cluster)
            w, y_pred, rmse = getResults(X_train,y_train,mu,sigma,ridge)
            train_rmses.append(np.array([cluster, ridge, sigma, rmse]))
            X = getGaussianBasis(X_val,mu,sigma)
            y_pred = getPolyfit(X,w)
            rmse = getRMSE(y_val,y_pred,ridge,w)
            val_rmses.append(np.array([cluster, ridge, sigma, rmse]))
            X = getGaussianBasis(X_test,mu,sigma)
            y_pred = getPolyfit(X,w)
            rmse = getRMSE(y_test,y_pred,ridge,w)
            test_rmses.append(np.array([cluster, ridge, sigma, rmse]))

# Finding best model based using validation set
val_rmses = np.array(val_rmses)
scores = val_rmses[:,-1].reshape(-1,1)
best_model = val_rmses[np.argmin(scores,axis=0)[0]]
if emrs :
    if ridge_go :
        np.savetxt("results/q7/ermsridgetrain"+str(X_train.shape[0])+".txt", train_rmses, fmt="%.2f")
        np.savetxt("results/q7/ermsridgetest"+str(X_train.shape[0])+".txt", test_rmses, fmt="%.2f")
        np.savetxt("results/q7/ermsridgeval"+str(X_train.shape[0])+".txt", val_rmses, fmt="%.2f")
    if go :
        np.savetxt("results/q7/ermstrain"+str(X_train.shape[0])+".txt", train_rmses, fmt="%.2f")
        np.savetxt("results/q7/ermstest"+str(X_train.shape[0])+".txt", test_rmses, fmt="%.2f")
        np.savetxt("results/q7/ermsval"+str(X_train.shape[0])+".txt", val_rmses, fmt="%.2f")

print("Parameters of best model are cluster, lambda, sigma",best_model[0],best_model[1],best_model[2])

cluster = int(best_model[0])
ridge = best_model[1]
sigma = best_model[2]

scatter_plot = 1
if scatter_plot :   
    mu = kmeans(X_train,cluster)
    w, y_pred, rmse = getResults(X_train,y_train,mu,sigma,ridge)
    fig = plt.figure()
    plt.scatter(y_train, y_pred, label="Train")
    w, y_pred, rmse = getResults(X_test,y_test,mu,sigma,ridge)
    plt.scatter(y_test, y_pred, label="Test")
    plt.xlabel("True Target")
    plt.ylabel("Model output")
    plt.title("Best model on train100")
    plt.legend()
    plt.savefig("results/q7scatter.png")
    plt.show()

function_plot = 0
if function_plot :
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    mu = kmeans(X_train,cluster)
    w, y_pred, rmse = getResults(X_train,y_train,mu,sigma,ridge)

    x = np.arange(mini[0],maxi[0],0.1)
    y = np.arange(mini[1],maxi[1],0.1)
    X, Y = np.meshgrid(x, y)
    X_data = []

    for x,y in zip(np.ravel(X), np.ravel(Y)) :
        X_data.append(np.array([x,y]))
    X_data = np.array(X_data)
    y_data = np.zeros(X_data.shape[0])

    zs = getGaussianBasis(X_data, mu, sigma)
    zs = getPolyfit(zs,w)
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z, alpha=0.9)
    ax.scatter(X_train[:,0],X_train[:,1],y_train,c='r')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    ax.set_title('Gaussian basis for 60 clusters, sigma 2')
    plt.savefig("results/q7function2000.png")
    plt.show()


