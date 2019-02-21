import numpy as np 
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal

np.random.seed(seed=42)

def getPrior(y) :
    unique, counts = np.unique(y, return_counts=True)
    print(unique)
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
            values =  multivariate_normal.pdf(X,mean=mu[class_val],cov=sigma)
        prob.append(value)
    return np.transpose(np.array(prob))

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

print("Computing prior")
classes, prior = getPrior(y_train)
print("Number of classes", len(classes))
print("Computing MLE mean")
means = np.array(getMLE(X_train, y_train)) 
lossfunction = np.array([[0,1,2],[1,0,1],[2,1,0]])
print("Computing class conditional density")
classConditional = getConditional(X_train, means, np.eye(2), "naive")
print("Computing risk")
risk = getRisk(lossfunction, classConditional, prior)

prediction = np.argmin(risk, axis=1)

unique, counts = np.unique(prediction, return_counts=True)

print("Accuracy of the prediction",np.sum(prediction == y_train)/y_train.shape[0])

# print(classConditional.shape)
# print(prior.shape)
