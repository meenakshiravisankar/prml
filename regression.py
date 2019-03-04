import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

def getPolyfit(X,w) :
    return np.matmul(X,w)

def getWeights(X,y,ridge) :
    return np.matmul(np.matmul(np.linalg.inv(ridge*np.eye(X.shape[1]) + np.matmul(np.transpose(X),X)),np.transpose(X)),y)

def getPolyfeatures(X,n) :
    return np.squeeze(np.transpose(np.array([X**i for i in range(n+1)])))

def getRMSE(ytrue,ypred,lamda=0,w=0) :
    return np.sqrt((np.sum(np.multiply(ytrue-ypred,ytrue-ypred)) + np.sum(np.multiply(w,w)))/ytrue.shape[0])
    

# Seeding
np.random.seed(10)

# Generate 100 points in (0,1)
X = np.random.uniform(low=0,high=1,size=100).reshape(-1,1)
y = np.exp(np.sin(2*np.pi*X)) + X + np.random.normal(0,0.2)

# splitting into train, test, validation - 70, 10, 20 and converting to numpy array
train_size = int(0.7*X.shape[0])
val_size = int(0.1*X.shape[0])
test_size = int(0.2*X.shape[0])
X_train = X[:train_size]
y_train = y[:train_size]
X_val = X[train_size:train_size+val_size]
y_val = y[train_size:train_size+val_size]
X_test = X[train_size+val_size:train_size+val_size+test_size]
y_test = y[train_size+val_size:train_size+val_size+test_size]

# Settings
degrees = [1,3,6,9]
ridges = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
sizes = [10,20,40,70]
all_degrees = np.linspace(0,9,10).astype(int)

# X_train_new = np.array([1,2,3,4])
# y_train_new = np.array([4,9,16,25])

fig = 221
rmses = []

default = 0
if default :
    ridges = [0]
    sizes = [10]
    figure = plt.figure()

for degree in degrees : 
    for ridge in ridges :
        for size in sizes : 
            X_train_new = X_train[:size]
            y_train_new = y_train[:size]
            X_train_new_poly = getPolyfeatures(X_train_new, degree)
            w = getWeights(X_train_new_poly, y_train_new, ridge)
            y_pred_new = getPolyfit(X_train_new_poly,w)
            rmse_train = getRMSE(y_train_new, y_pred_new)
            rmse_val = getRMSE(y_val,getPolyfit(getPolyfeatures(X_val,degree),w))
            rmses.append(np.array([degree, ridge, size, rmse_val]))

            if default :
                plt.subplot(fig)
                fig+=1
                if fig%2 :
                    plt.xlabel("x")
                else :
                    plt.ylabel("target")
                plt.plot(X_train_new,y_train_new,'*')
                x = np.linspace(0,1,100)
                x_poly = getPolyfeatures(x,degree)
                y_func = getPolyfit(x_poly,w)
                np.savetxt("results/coefficients/"+str(degree)+".txt",w,newline=" ")
                M = "M="+str(degree)
                plt.plot(x, y_func,label=M)
                plt.legend()

                figure.suptitle("Data, Target, Regression Output")
                plt.savefig("results/targetvsx.png")

best_model = rmses[np.argmin(np.array(rmses),axis=0)[-1]]
print("Parameters of best model has degree {:.2f}, lambda {:.2f}, train size {:.2f}".format(best_model[0],best_model[1],best_model[2]))

rmse1 = []
deg1 = []
rmse2 = []
deg2 = []

for rmse in rmses :
    if rmse[2] == 10 and rmse[1] !=0 :
        rmse1.append(rmse[3])
        deg1.append(rmse[0])
    if rmse[1] == 0 :
        rmse2.append(rmse[3])
        deg2.append(rmse[0])

plt.rcParams.update({'font.size': 8})

plt.subplot(221)
xaxis = np.log(np.array(ridges[1:]))
for i in range(int(len(rmse1)/xaxis.shape[0])) :
    start = i*xaxis.shape[0]
    end = start+xaxis.shape[0]
    yaxis = rmse1[start : end]
    plt.plot(xaxis,yaxis,'-*',label="M="+str(degrees[i]))

plt.xlabel(r"$\ln  \lambda $",fontsize=6)
plt.ylabel(r"$E_{RMS}$",fontsize=8)
plt.title("RMS Error vs Ridge Parameter (Train Size 10)",fontsize=8)
plt.legend(fontsize=8)

plt.subplot(222)
xaxis = np.array(sizes)
for i in range(int(len(rmse2)/xaxis.shape[0])) :
    start = i*xaxis.shape[0]
    end = start+xaxis.shape[0]
    yaxis = rmse2[start : end]
    plt.plot(xaxis,yaxis,'-*',label="M="+str(degrees[i]))
plt.xlabel("Train Size",fontsize=8)
plt.ylabel(r"$E_{RMS}$",fontsize=8)
plt.title("RMS Error vs Train Size (No reg)",fontsize=8)
plt.legend(fontsize=8)

plt.subplot(223)

size = best_model[2].astype(int)
ridge = best_model[1]
degree = best_model[0].astype(int)

X_train_new = X_train[:size]
y_train_new = y_train[:size]
X_train_new_poly = getPolyfeatures(X_train_new, degree)
w = getWeights(X_train_new_poly, y_train_new, ridge)
y_pred_new = getPolyfit(X_train_new_poly,w)
y_pred_test = getPolyfit(getPolyfeatures(X_test,degree),w)

plt.plot(y_train_new, y_pred_new,"*",label="Train")
plt.plot(y_test, y_pred_test,"+",label="Test")
plt.xlabel("True target",fontsize=8)
plt.ylabel("Model output",fontsize=8)
plt.legend(fontsize=8)

ridge = 0
size = 70
for degree in all_degrees : 
    X_train_new = X_train[:size]
    y_train_new = y_train[:size]
    X_train_new_poly = getPolyfeatures(X_train_new, degree)
    w = getWeights(X_train_new_poly, y_train_new, ridge)
    y_pred_new = getPolyfit(X_train_new_poly,w)
    rmse_train = getRMSE(y_train_new, y_pred_new)
    rmse_test = getRMSE(y_test,getPolyfit(getPolyfeatures(X_test,degree),w))

plt.subplot(224)
plt.plot(all_degress,rmse_train,"-*",label="Train")
plt.plot(all_degrees,rmse_test,"-*",label="Test")
plt.savefig("results/erms.png")
plt.show()
