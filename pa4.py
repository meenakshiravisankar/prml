import numpy as np

# read dataset
data = np.loadtxt("../Datasets_PRML_A1/Dataset_5_Team_39.csv", delimiter=',', dtype=None)

# number of data points
n = data.size

# sum of all the data points
Sn = data.sum()

mle_mean = Sn / n
print(mle_mean)

mu_n = []

for factor in [0.1, 1, 10, 100]:
    temp = (Sn - factor**2) / (n + factor**2)
    mu_n.append(temp)

print(mu_n)
