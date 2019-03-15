import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# seeding
np.random.seed(42)

# function to evaluate normal function
def eval1DGaussian(X, mu, sigma):
    power = (np.sum((X-mu)*(X-mu)) * -1) / (2 * sigma * sigma)
    factor = 1/(np.sqrt(2 * np.pi) * sigma)
    value = factor * np.exp(power)
    return value

# read dataset
data = np.loadtxt("../Datasets_PRML_A1/Dataset_5_Team_39.csv", delimiter=',', dtype=None)
mu0 = -1

# fixing the x-axis limits for the plots
x_plot = np.linspace(-25, 25, 1000)

# range of values taken by (sigma^2/sigma_0^2)
factors = [0.1, 1, 10, 100]

# range of dataset sizes
dataset_sizes = [10, 100, 1000]

fig = 221

for factor in factors:
    mu_n = []
    sigma_estimates = []

    plt.subplot(fig)
    plt.tight_layout()
    plt.title(r'$\frac{\sigma^2}{\sigma^2_{0}}=$'+str(factor),fontsize=8)
    fig+=1

    for n in dataset_sizes:
        np.random.shuffle(data)
        X = data[0:n]
        Sn = sum(X)
        Sn2 = sum(X*X)

        # posterior mean of mu
        mu = (Sn + mu0*factor) / (n + factor)
        mu_n.append(mu)

        # ML estimate of sigma
        sigma = (Sn2 - 2*Sn*mu + mu*mu) / n
        sigma_estimates.append(sigma)

    # plotting the posterior density
    for i in range(3):
        labelname = "n = " + str(dataset_sizes[i])
        plt.plot(x_plot, stats.norm.pdf(x_plot, mu_n[i], sigma_estimates[i]), label = labelname)
        plt.ylim(top=0.4)
        if fig == 223 :
            plt.legend(fontsize=8, loc='upper right')

plt.suptitle("Estimated Density P(x/D)",fontsize=8)
plt.savefig("results/q4.png")
plt.show()
