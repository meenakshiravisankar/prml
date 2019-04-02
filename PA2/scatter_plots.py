import numpy as np
import matplotlib.pyplot as plt

f = plt.figure(1)
data1 = np.loadtxt("../Datasets_PRML_A2/Dataset_1_Team_39.csv", delimiter=',', dtype=None)
plt.scatter(data1[:,0],data1[:,1],c=data1[:,2])
plt.savefig("results/Dataset1_scatter")

f = plt.figure(2)
data2 = np.loadtxt("../Datasets_PRML_A2/Dataset_2_Team_39.csv", delimiter=',', dtype=None)
plt.scatter(data2[:,0],data2[:,1],c=data2[:,2])
plt.savefig("results/Dataset2_scatter")

f = plt.figure(3)
data3 = np.loadtxt("../Datasets_PRML_A2/Dataset_3_Team_39.csv", delimiter=',', dtype=None)
plt.scatter(data3[:,0],data3[:,1],c=data3[:,2])
plt.savefig("results/Dataset3_scatter")

f = plt.figure(4)
data4 = np.loadtxt("../Datasets_PRML_A2/Dataset_4_Team_39.csv", delimiter=',', dtype=None)
plt.scatter(data4[:,0],data4[:,1],c=data4[:,2])
plt.savefig("results/Dataset4_scatter")

f = plt.figure(5)
data5 = np.loadtxt("../Datasets_PRML_A2/Dataset_5_Team_39.csv", delimiter=',', dtype=None)
plt.scatter(data5[:,0],data5[:,1],c=data5[:,2])
plt.savefig("results/Dataset5_scatter")
