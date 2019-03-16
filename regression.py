import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from scipy import linalg
import warnings
from sklearn.model_selection import train_test_split

# warnings.filterwarnings("ignore")

def getFunc(X) :
    return np.exp(np.sin(2*np.pi*X)) + X

def getPolyfit(X,w) :
    return np.matmul(X,w)

def getWeights(X,y,ridge) :
    # A = np.matmul(np.transpose(X),X)
    # Y = np.matmul(np.transpose(X),y)
    # w = linalg.solve(A,Y)
    # return w
    return np.matmul(linalg.inv(ridge*np.eye(X.shape[1]) + np.matmul(np.transpose(X),X)), np.matmul(np.transpose(X),y))

def getPolyfeatures(X,n) :
    return np.squeeze(np.transpose(np.array([X**i for i in range(n+1)]))).reshape(-1,n+1)

def getRMSE(ytrue,ypred,ridge,w) :
    return np.sqrt((np.sum(np.multiply(ytrue-ypred,ytrue-ypred)) + ridge*np.sum(np.multiply(w,w)))/ytrue.shape[0])
    
def getResults(X,y,degree,ridge) :
    X = getPolyfeatures(X, degree)
    w = getWeights(X, y, ridge)
    y_pred = getPolyfit(X,w)
    rmse = getRMSE(y,y_pred,ridge,w)
    return w, y_pred, rmse

# Seeding
np.random.seed(42)

# Initialising to float64 for precision
X = np.ndarray(shape=(100,1),dtype=np.float64)
y = np.ndarray(shape=(100,1),dtype=np.float64)

# Generate 100 points in (0,1)
X = np.random.uniform(low=0,high=1,size=100).reshape(100,1)
noise = np.random.normal(0,np.sqrt(0.2),100).reshape(100,1)
func = getFunc(X)
y = func + noise

# Shuffling data
data = np.append(X,y,axis=1)
np.random.shuffle(data)
X = data[:,0]
y = data[:,1]

# splitting into train, test, validation - 70, 20, 10
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
ridges = [10**(-13+x) for x in range(14)]
ridges[0] = 0
sizes = [10,20,30,40,50,60,70]
all_degrees = np.linspace(0,9,10).astype(int)


fig = 221
rmses = []

# set default to 1 if plots needed without regularisation
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
            # X_train_new_poly = getPolyfeatures(X_train_new, degree)
            # w = getWeights(X_train_new_poly, y_train_new, ridge)
            # y_pred_new = getPolyfit(X_train_new_poly,w)
            # rmse_train = getRMSE(y_train_new, y_pred_new, ridge, w)
            w, y_pred_new, rmse = getResults(X_train_new, y_train_new, degree, ridge)
            rmse_val = getRMSE(y_val,getPolyfit(getPolyfeatures(X_val,degree),w), ridge, w)
            rmses.append(np.array([degree, ridge, size, rmse_val]))

            if default :
                plt.subplot(fig)
                fig+=1
                if fig%2 :
                    plt.xlabel("x")
                else :
                    plt.ylabel("target")

                # Training data
                plt.plot(X_train_new,y_train_new,'.')
                
                # Curve fit
                x = np.linspace(0,1,100)
                y_func = getFunc(x)
                plt.plot(x,y_func,label="target")
                x_poly = getPolyfeatures(x,degree)
                y_func = getPolyfit(x_poly,w)
                # Coefficients
                np.savetxt("results/coefficients/noreg/"+str(degree)+".txt",w,newline=" ")
                M = "M="+str(degree)
                plt.plot(x, y_func,label=M)
                plt.xlim(0,1)
                plt.ylim(0,3)
                plt.legend()
                figure.suptitle("Data, Target, Regression Output (without regularisation)")
                plt.savefig("results/targetvsx.png")

# Finding best model based using validation set
rmses = np.array(rmses)
scores = rmses[:,-1].reshape(-1,1)
best_model = rmses[np.argmin(scores,axis=0)[0]]
print("Parameters of best model are degree, lambda, train size",best_model[0],best_model[1],best_model[2])

if default :
    raise SystemExit

rmse1 = []
deg1 = []
rmse2 = []
deg2 = []

# rmse1 - with ridge regularisation and train size 10, rmse2 - without ridge regression and vary train size
mini1 = 10
mini2 = 10
for rmse in rmses :
    if rmse[2] == 10 and rmse[1] !=0 :
        rmse1.append(rmse[3])
        deg1.append(rmse[0])
        if mini1 > rmse[3] :
            mini1 = rmse[3]
            best_ridge = rmse[1]
    if rmse[1] == 0 :
        rmse2.append(rmse[3])
        deg2.append(rmse[0])
        if mini2 > rmse[3] :
            mini2 = rmse[3]
            best_size = rmse[2]


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
plt.title("Val RMS Error vs Ridge Parameter (Train Size 10)",fontsize=8)
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
plt.title("Val RMS Error vs Train Size (No reg)",fontsize=8)
plt.legend(fontsize=8, loc="upper right")

plt.subplot(223)
# Target vs output plot
size = best_model[2].astype(int)
ridge = best_model[1]
degree = best_model[0].astype(int)

X_train_new = X_train[:size]
y_train_new = y_train[:size]
w, y_pred_new, rmse = getResults(X_train_new, y_train_new, degree, ridge)
y_pred_test = getPolyfit(getPolyfeatures(X_test,degree),w)

plt.plot(y_train_new, y_pred_new,"*",label="Train")
plt.plot(y_test, y_pred_test,"+",label="Test")
plt.xlabel("True target",fontsize=8)
plt.ylabel("Model output",fontsize=8)
plt.legend(fontsize=8)


# Plotting Erms for train and test data without any regularisation
ridge = 0
size = 10
X_train_new = X_train[:size]
y_train_new = y_train[:size]

rmse_train = []
rmse_test = [] 

for degree in all_degrees : 
    w, y_pred_new, rmse = getResults(X_train_new, y_train_new, degree, ridge)
    rmse_train.append(rmse)
    rmse_test.append(getRMSE(y_test,getPolyfit(getPolyfeatures(X_test,degree),w), ridge, w))

plt.subplot(224)
plt.tight_layout()
plt.plot(all_degrees,rmse_train,"-*",label="Train")
plt.plot(all_degrees,rmse_test,"-*",label="Test")
plt.legend(fontsize=8,loc="upper right")
plt.xlabel("Degree",fontsize=8)
plt.ylabel(r"$E_{RMS}$",fontsize=8)
plt.savefig("results/erms.png")


# Analysing overfitting case
figure = plt.figure()
degree = 9
size = 10

plt.subplot(221)
x = np.linspace(0,1,100)
x_poly = getPolyfeatures(x,degree)
# without regularisation for 9th degree polynomial
X_train_new = X_train[:10]
y_train_new = y_train[:10]
plt.plot(X_train_new,y_train_new,'.')
w, y_pred_new, rmse = getResults(X_train_new, y_train_new, degree, 0)
y_func = getPolyfit(x_poly,w)
plt.plot(x, y_func,label="output")
plt.title("No regularisation")
y_func = getFunc(x)
plt.plot(x,y_func,label="target")
plt.tight_layout()
plt.ylabel("y")
plt.legend()

plt.subplot(223)

plt.plot(X_train_new,y_train_new,'.')
# with ridge regression for 9th degree polynomial
X_train_new = X_train[:size]
y_train_new = y_train[:size]
w, y_pred_new, rmse = getResults(X_train_new, y_train_new, degree, best_ridge)
y_func = getPolyfit(x_poly,w)
np.savetxt("results/coefficients/lowreg/"+str(degree)+".txt",w,newline=" ")
plt.plot(x, y_func,label="output")
plt.title(r'$\lambda=$'+str(best_ridge))
y_func = getFunc(x)
plt.plot(x,y_func,label="target")
plt.tight_layout()
plt.ylabel("y")
plt.xlabel("x")
plt.legend()

plt.subplot(222)
# with regularisation using large train size for 9th degree polynomial
X_train_new = X_train[:int(best_size)]
y_train_new = y_train[:int(best_size)]
plt.plot(X_train_new,y_train_new,'.')
w, y_pred_new, rmse = getResults(X_train_new, y_train_new, degree, 0)
y_func = getPolyfit(x_poly,w)
plt.plot(x, y_func,label="output")
plt.title("Train size - "+str(int(best_size)))
y_func = getFunc(x)
plt.plot(x,y_func,label="target")
plt.tight_layout()
plt.legend()



plt.subplot(224)
# high regularisation for 9th degree polynomial
X_train_new = X_train[:10]
y_train_new = y_train[:10]
plt.plot(X_train_new,y_train_new,'.')
w, y_pred_new, rmse = getResults(X_train_new, y_train_new, degree, 1)
y_func = getPolyfit(x_poly,w)
np.savetxt("results/coefficients/highreg/"+str(degree)+".txt",w,newline=" ")
plt.plot(x, y_func,label="output")
plt.title(r'$\lambda=$'+str(1))
plt.xlabel("x")
y_func = getFunc(x)
plt.plot(x,y_func,label="target")
plt.tight_layout()
plt.legend()

# figure.suptitle("Analysis of Overfitting for degree 9")
plt.savefig("results/fitting.png")


plt.show()