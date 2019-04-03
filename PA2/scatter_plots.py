import numpy as np
import matplotlib.pyplot as plt

classes = [0,1]
f = plt.figure(1)
data1 = np.loadtxt("../Datasets_PRML_A2/Dataset_1_Team_39.csv", delimiter=',', dtype=None)
for class_val in classes :
    idx = (data1[:,2]==class_val)
    plt.scatter(data1[idx,0],data1[idx,1],label="class "+str(class_val),alpha=0.7)
plt.legend()
plt.savefig("results/Dataset1_scatter")

f = plt.figure(2)
data2 = np.loadtxt("../Datasets_PRML_A2/Dataset_2_Team_39.csv", delimiter=',', dtype=None)
for class_val in classes :
    idx = (data2[:,2]==class_val)
    plt.scatter(data2[idx,0],data2[idx,1],label="class "+str(class_val),alpha=0.7)
plt.legend()
plt.savefig("results/Dataset2_scatter")

f = plt.figure(3)
data3 = np.loadtxt("../Datasets_PRML_A2/Dataset_3_Team_39.csv", delimiter=',', dtype=None)
for class_val in classes :
    idx = (data3[:,2]==class_val)
    plt.scatter(data3[idx,0],data3[idx,1],label="class "+str(class_val),alpha=0.7)
plt.legend()
plt.savefig("results/Dataset3_scatter")

f = plt.figure(4)
data4 = np.loadtxt("../Datasets_PRML_A2/Dataset_4_Team_39.csv", delimiter=',', dtype=None)
for class_val in classes :
    idx = (data4[:,2]==class_val)
    plt.scatter(data4[idx,0],data4[idx,1],label="class "+str(class_val),alpha=0.7)
plt.legend()
plt.savefig("results/Dataset4_scatter")


f = plt.figure(5)
data5 = np.loadtxt("../Datasets_PRML_A2/Dataset_5_Team_39.csv", delimiter=',', dtype=None)
for class_val in classes :
    idx = (data5[:,2]==class_val)
    plt.scatter(data5[idx,0],data5[idx,1],label="class "+str(class_val),alpha=0.7)
plt.legend()
plt.savefig("results/Dataset5_scatter")
