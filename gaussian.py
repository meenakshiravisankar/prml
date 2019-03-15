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

data1 = np.loadtxt("../Datasets_PRML_A1/train100.txt", delimiter=' ', dtype=None)
data2 = np.loadtxt("../Datasets_PRML_A1/val.txt", delimiter=' ', dtype=None)
data3 = np.loadtxt("../Datasets_PRML_A1/test.txt", delimiter=' ', dtype=None)

np.random.shuffle(data1)

X_train = data1[:,:2]
y_train = data1[:,-1]
X_val = data2[:,:2]
y_val = data2[:,-1]
X_test = data3[:,:2]
y_test = data3[:,-1]

# Settings 
clusters = [2,4,10,20,30,40,50,60]
ridges = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
sigmas = [0.01,0.1,1,2]
ridges[0] = 0

default = 1
if default :
    clusters = [60]
    ridges = [0]
    sigmas = [1]

rmses = []
mini = np.min(X_train,axis=0)-1
maxi = np.max(X_train,axis=0)+1

for cluster in clusters :
    for ridge in ridges :
        for sigma in sigmas :
            mu = kmeans(X_train,cluster)
            w, y_pred, rmse = getResults(X_train,y_train,mu,sigma,ridge)
            print(rmse)

x = np.linspace(mini[0],maxi[0],1000)
x = np.random.uniform(mini[0],maxi[0],1000)
y = np.linspace(mini[1],maxi[1],1000)
y = np.linspace(mini[1],maxi[1],1000)

x_train = np.array([x,y]).reshape(1000,-1)
x_train = getGaussianBasis(x_train,mu,sigma)
y_pred =  getPolyfit(x_train,w)
# Axes3D.plot_surface(X=x,Y=y,Z=y_pred)
# plt.show()
fig = plt.figure()
ax = fig.gca(projection='3d')
print(x.shape, y.shape, y_pred.shape)
ax.plot_wireframe(x, y, y_pred)
ax.scatter(X_train[:,0],X_train[:,1],y_train,c='r')
plt.show()







# print(y_pred)
# print(y_train)
# plt.scatter(y_pred, y_train)
# plt.show()
# fig = 221
# rmses = []

# # set default to 1 if plots needed without regularisation
# default = 1
# if default :
#     ridges = [0]
#     figure = plt.figure()

# for cluster in clusters : 
#     for ridge in ridges :
#             X_train_new = X_train            
#             y_train_new = y_train
#             # X_train_new_poly = getPolyfeatures(X_train_new, degree)
#             # w = getWeights(X_train_new_poly, y_train_new, ridge)
#             # y_pred_new = getPolyfit(X_train_new_poly,w)
#             # rmse_train = getRMSE(y_train_new, y_pred_new, ridge, w)
#             w, y_pred_new, rmse = getResults(X_train_new, y_train_new, degree, ridge)
#             rmse_val = getRMSE(y_val,getPolyfit(getPolyfeatures(X_val,degree),w), ridge, w)
#             rmses.append(np.array([degree, ridge, size, rmse_val]))

#             if default :
#                 plt.subplot(fig)
#                 fig+=1
#                 if fig%2 :
#                     plt.xlabel("x")
#                 else :
#                     plt.ylabel("target")

#                 # Training data
#                 plt.plot(X_train_new,y_train_new,'.')
                
#                 # Curve fit
#                 x = np.linspace(0,1,100)
#                 y_func = getFunc(x)
#                 plt.plot(x,y_func,label="target")
#                 x_poly = getPolyfeatures(x,degree)
#                 y_func = getPolyfit(x_poly,w)
#                 # Coefficients
#                 np.savetxt("results/q7/coefficients/noreg/"+str(degree)+".txt",w,newline=" ")
#                 M = "M="+str(degree)
#                 plt.plot(x, y_func,label=M)
#                 plt.xlim(0,1)
#                 plt.ylim(0,3)
#                 plt.legend()
#                 figure.suptitle("Data, Target, Regression Output (without regularisation)")
#                 plt.savefig("results/q7/targetvsx.png")

# # Finding best model based using validation set
# rmses = np.array(rmses)
# scores = rmses[:,-1].reshape(-1,1)
# best_model = rmses[np.argmin(scores,axis=0)[0]]
# print("Parameters of best model has degree, lambda, train size",best_model[0],best_model[1],best_model[2])

# if default :
#     raise SystemExit

# rmse1 = []
# deg1 = []
# rmse2 = []
# deg2 = []

# # rmse1 - with ridge regularisation and train size 10, rmse2 - without ridge regression and vary train size
# mini1 = 10
# mini2 = 10
# for rmse in rmses :
#     if rmse[2] == 10 and rmse[1] !=0 :
#         rmse1.append(rmse[3])
#         deg1.append(rmse[0])
#         if mini1 > rmse[3] :
#             mini1 = rmse[3]
#             best_ridge = rmse[1]
#     if rmse[1] == 0 :
#         rmse2.append(rmse[3])
#         deg2.append(rmse[0])
#         if mini2 > rmse[3] :
#             mini2 = rmse[3]
#             best_size = rmse[2]


# plt.rcParams.update({'font.size': 8})
# plt.subplot(221)
# xaxis = np.log(np.array(ridges[1:]))
# for i in range(int(len(rmse1)/xaxis.shape[0])) :
#     start = i*xaxis.shape[0]
#     end = start+xaxis.shape[0]
#     yaxis = rmse1[start : end]
#     plt.plot(xaxis,yaxis,'-*',label="M="+str(degrees[i]))

# plt.xlabel(r"$\ln  \lambda $",fontsize=6)
# plt.ylabel(r"$E_{RMS}$",fontsize=8)
# plt.title("Val RMS Error vs Ridge Parameter (Train Size 10)",fontsize=8)
# plt.legend(fontsize=8)


# plt.subplot(222)
# xaxis = np.array(sizes)
# for i in range(int(len(rmse2)/xaxis.shape[0])) :
#     start = i*xaxis.shape[0]
#     end = start+xaxis.shape[0]
#     yaxis = rmse2[start : end]
#     plt.plot(xaxis,yaxis,'-*',label="M="+str(degrees[i]))
# plt.xlabel("Train Size",fontsize=8)
# plt.ylabel(r"$E_{RMS}$",fontsize=8)
# plt.title("Val RMS Error vs Train Size (No reg)",fontsize=8)
# plt.legend(fontsize=8, loc="upper right")

# plt.subplot(223)
# # Target vs output plot
# size = best_model[2].astype(int)
# ridge = best_model[1]
# degree = best_model[0].astype(int)

# X_train_new = X_train[:size]
# y_train_new = y_train[:size]
# w, y_pred_new, rmse = getResults(X_train_new, y_train_new, degree, ridge)
# y_pred_test = getPolyfit(getPolyfeatures(X_test,degree),w)

# plt.plot(y_train_new, y_pred_new,"*",label="Train")
# plt.plot(y_test, y_pred_test,"+",label="Test")
# plt.xlabel("True target",fontsize=8)
# plt.ylabel("Model output",fontsize=8)
# plt.legend(fontsize=8)


# # Plotting Erms for train and test data without any regularisation
# ridge = 0
# size = 10
# X_train_new = X_train[:size]
# y_train_new = y_train[:size]

# rmse_train = []
# rmse_test = [] 

# for degree in all_degrees : 
#     w, y_pred_new, rmse = getResults(X_train_new, y_train_new, degree, ridge)
#     rmse_train.append(rmse)
#     rmse_test.append(getRMSE(y_test,getPolyfit(getPolyfeatures(X_test,degree),w), ridge, w))

# plt.subplot(224)
# plt.tight_layout()
# plt.plot(all_degrees,rmse_train,"-*",label="Train")
# plt.plot(all_degrees,rmse_test,"-*",label="Test")
# plt.legend(fontsize=8,loc="upper right")
# plt.xlabel("Degree",fontsize=8)
# plt.ylabel(r"$E_{RMS}$",fontsize=8)
# plt.savefig("results/q7/erms.png")


# # Analysing overfitting case
# figure = plt.figure()
# degree = 9
# size = 10

# plt.subplot(221)
# x = np.linspace(0,1,100)
# x_poly = getPolyfeatures(x,degree)
# # without regularisation for 9th degree polynomial
# X_train_new = X_train[:10]
# y_train_new = y_train[:10]
# plt.plot(X_train_new,y_train_new,'.')
# w, y_pred_new, rmse = getResults(X_train_new, y_train_new, degree, 0)
# y_func = getPolyfit(x_poly,w)
# plt.plot(x, y_func,label="output")
# plt.title("No regularisation")
# y_func = getFunc(x)
# plt.plot(x,y_func,label="target")
# plt.tight_layout()
# plt.ylabel("y")
# plt.legend()

# plt.subplot(223)

# plt.plot(X_train_new,y_train_new,'.')
# # with ridge regression for 9th degree polynomial
# X_train_new = X_train[:size]
# y_train_new = y_train[:size]
# w, y_pred_new, rmse = getResults(X_train_new, y_train_new, degree, best_ridge)
# y_func = getPolyfit(x_poly,w)
# np.savetxt("results/q7/coefficients/lowreg/"+str(degree)+".txt",w,newline=" ")
# plt.plot(x, y_func,label="output")
# plt.title(r'$\lambda=$'+str(best_ridge))
# y_func = getFunc(x)
# plt.plot(x,y_func,label="target")
# plt.tight_layout()
# plt.ylabel("y")
# plt.xlabel("x")
# plt.legend()

# plt.subplot(222)
# # with regularisation using large train size for 9th degree polynomial
# X_train_new = X_train[:int(best_size)]
# y_train_new = y_train[:int(best_size)]
# plt.plot(X_train_new,y_train_new,'.')
# w, y_pred_new, rmse = getResults(X_train_new, y_train_new, degree, 0)
# y_func = getPolyfit(x_poly,w)
# plt.plot(x, y_func,label="output")
# plt.title("Train size - "+str(int(best_size)))
# y_func = getFunc(x)
# plt.plot(x,y_func,label="target")
# plt.tight_layout()
# plt.legend()



# plt.subplot(224)
# # high regularisation for 9th degree polynomial
# X_train_new = X_train[:10]
# y_train_new = y_train[:10]
# plt.plot(X_train_new,y_train_new,'.')
# w, y_pred_new, rmse = getResults(X_train_new, y_train_new, degree, 1)
# y_func = getPolyfit(x_poly,w)
# np.savetxt("results/q7/coefficients/highreg/"+str(degree)+".txt",w,newline=" ")
# plt.plot(x, y_func,label="output")
# plt.title(r'$\lambda=$'+str(1))
# plt.xlabel("x")
# y_func = getFunc(x)
# plt.plot(x,y_func,label="target")
# plt.tight_layout()
# plt.legend()

# # figure.suptitle("Analysis of Overfitting for degree 9")
# plt.savefig("results/q7/fitting.png")


# plt.show()