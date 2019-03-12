import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

np.random.seed(42)

def eval1DGaussian(X, mu, sigma):
    power = (np.sum((X-mu)*(X-mu)) * -1) / (2 * sigma * sigma)
    factor = 1/(np.sqrt(2 * np.pi) * sigma)
    value = factor * np.exp(power)
    return value

# read dataset
data = np.loadtxt("../Datasets_PRML_A1/Dataset_5_Team_39.csv", delimiter=',', dtype=None)
mu0 = -1

x_plot = np.linspace(-25, 25, 1000)
factors = [0.1, 1, 10, 100]
dataset_sizes = [10, 100, 1000]

for factor in factors:
    mu_n = []
    sigma_estimates = []
    for n in dataset_sizes:
        np.random.shuffle(data)
        X = data[0:n]
        Sn = sum(X)
        Sn2 = sum(X*X)
        mu = (Sn + mu0*factor) / (n + factor)
        mu_n.append(mu)

        # Now that we know mu, we can do either ML estimate or Bayesian estimate of sigma.
        # ML estimate of sigma is simply (Sn2 - 2*Sn*mu + mu*mu) / n .
        # After doing Bayesian estimate with MLE of sigma as the prior, we get posterior to be:
        sigma = (Sn2 - 2*Sn*mu + mu*mu) / (n + 2)
        # This is almost same as ML estimate, except that it is n+2 in the denominator, instead of n.
        # So for large n, this doesn't make any difference.
        # Of course, one reason for this is that we chose a uniform prior for sigma.
        # I chose the ML estimate of sigma as its uniform prior for Bayesian estimation for convenience :P
        sigma_estimates.append(sigma)
    for i in range(3):
        labelname = "n = " + str(dataset_sizes[i])
        plt.plot(x_plot, stats.norm.pdf(x_plot, mu_n[i], sigma_estimates[i]), label = labelname)
    plt.legend()
    plt.show()
