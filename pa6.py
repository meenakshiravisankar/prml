import numpy as np
import functions as f
import math
import random
import matplotlib.pyplot as plt

random.seed(42)

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
    # find w using (phi(x)' * phi(x))^-1 * phi(x)' * y
    # here, I have put some dummy
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
