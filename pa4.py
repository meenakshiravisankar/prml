import numpy as np

np.random.seed(42)

# read dataset
data = np.loadtxt("../Datasets_PRML_A1/Dataset_5_Team_39.csv", delimiter=',', dtype=None)

# number of data points
for n in [10, 100, 1000]:
    np.random.shuffle(data)
    X = data[0:n]
    Sn = sum(X)
    Sn2 = sum(X*X)

    # estimate of sigma assuming mu = -1 turns out to be (n + 2)/SIGMA(Xi + 1)^2
    sigma_estimate = (n + 2) / (Sn2 + 2*Sn + 1)
    print("sigma estimate:", sigma_estimate)

    mu_n = []

    for factor in [0.1, 1, 10, 100]:
        temp = (Sn - factor**2) / (n + factor**2)
        mu_n.append(temp)

    print("mu_n:", mu_n)
