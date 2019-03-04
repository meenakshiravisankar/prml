# Reference for plot: https://pythonspot.com/matplotlib-scatterplot/
import numpy as np
import functions as f
import math
import random
import matplotlib.pyplot as plt

random.seed(42)
colors = (0,0,0)
area = np.pi*3

data = np.genfromtxt("../Datasets_PRML_A1/train100.txt", delimiter=',', dtype=None, encoding=None)
data = [list(np.fromstring(data[i], dtype=float, count=3, sep=' ')) for i in range(len(data))]

w = []
# find w using (phi(x)' * phi(x))^-1 * phi(x)' * y
# plot to show the overfitting

# controlling the overfitting by using Ridge Regression
lambdas = [0.0001, 0.001, 0.01, 0.1, 1, 10]
# find w taking lambda into account and plot
# for lb in lambdas:
#     plt.scatter(terget_values, predicted_values, s=area, c=colors, alpha=0.5)
#     plt.title('Scatter plot for lambda = %f' %(lb))
#     plt.xlabel('target')
#     plt.ylabel('predicted')
#     plt.show()

# controlling the overfitting by increasing the training data size
data_1 = np.genfromtxt("../Datasets_PRML_A1/train1000.txt", delimiter=',', dtype=None, encoding=None)
data_1 = [list(np.fromstring(data_1[i], dtype=float, count=3, sep=' ')) for i in range(len(data_1))]

# find w and plot
