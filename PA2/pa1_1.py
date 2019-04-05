import numpy as np
import matplotlib.pyplot as plt

# read dataset
data = np.loadtxt("../Datasets_PRML_A2/Dataset_4_Team_39.csv", delimiter=',', dtype=None)

# shuffling
np.random.shuffle(data)

# splitting into train, test as 65% and 35%
train_size = int(0.65*data.shape[0])
test_size = int(0.35*data.shape[0])


# splitting into train and val sets from total train data

val_size = int(0.2*train_size)
train_size = int(0.8*train_size)

X_train = data[:train_size,:2]
y_train = data[:train_size,-1]

X_val = data[train_size:train_size+val_size,:2]
y_val = data[train_size:train_size+val_size,-1]

X_test = data[train_size+val_size:train_size+val_size+test_size,:2]
y_test = data[train_size+val_size:train_size+val_size+test_size,-1]

print("Size of train, validation and test sets",X_train.shape,X_val.shape,X_test.shape)

print(train_size, val_size, test_size, data.shape)

