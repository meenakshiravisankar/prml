import numpy as np
import functions as f
import math
import random
import matplotlib.pyplot as plt

random.seed(42)

'''Let X be unif(-1; 1) and Y = e(tanh(2p*i*x)) - x. Let the noise be N(0; sigma2), with sigma2 = 0.2.
Analyze bias-variance trade-off for the three models corresponding to the polynomials
of degrees 1, 5, 9 respectively. In particular do the following for each model:
1. Sample 10 points and do Ridge regression for lambda = 0.001, 0.01, 0.1.
2. Store the empirical risk observed for this data-set.
Repeat the steps 1,2 above for 1000 times. Plot the histogram of the empirical risk for
each of the models.
Interpret the results, and relate the observed behaviour with bias-variance trade-off.
Refer to Figure 9.4 of Duda's book.'''

def func(x):
    return math.exp(2*math.pi*x) - x

def noise(x):
    return f.eval1DGaussian(0, math.sqrt(0.2))

def eval(w, x):
    sum = 0
    for i in range(len(w)):
        sum += (w[i] * (x**i))
    return sum

def norm(w):
    norm_val = 0
    for i in range(len(w)):
        norm_val += w[i]*w[i]
    return norm_val

degrees = [1, 5, 9]
lambdas = [0.001, 0.01, 0.1]
for d in degrees:
    w = []
    for i in range(d):
        w.append(random.randint(1,101))
    for myLambda in lambdas:
        sums_y = []
        for i in range(1000):
            samples = []
            for i in range(10):
                samples.append(2 * np.random.random_sample() - 1)
            for i in range(10):
                sum = 0
                temp = eval(w, samples[i])
                sum += temp*temp
                sum = sum/2.0
                sum += myLambda*norm(w)
                sums_y.append(sum)
    plt.hist(sums_y, bins=10,alpha=0.5)
    plt.show()
