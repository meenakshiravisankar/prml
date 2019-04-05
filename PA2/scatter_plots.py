import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

data1 = np.loadtxt("../Datasets_PRML_A2/Dataset_1_Team_39.csv", delimiter=',', dtype=None)
classes = np.array(np.unique(data1[:,2], return_counts=False),dtype=int)

for class_val in classes :
    idx = (data1[:,2]==class_val)
    plt.scatter(data1[idx,0],data1[idx,1],label="class "+str(class_val)+" - "+str(np.sum(idx)),s=10)
    plt.title("Dataset 1")
plt.legend()
plt.savefig("results/Dataset1_scatter")

# plt.subplot(231)
# data2 = np.loadtxt("../Datasets_PRML_A2/Dataset_2_Team_39.csv", delimiter=',', dtype=None)
# for class_val in classes :
#     idx = (data2[:,2]==class_val)
#     plt.scatter(data2[idx,0],data2[idx,1],label="class "+str(class_val),s=10)
# plt.tight_layout()

# # plt.legend()
# # # plt.savefig("results/Dataset2_scatter")

# plt.subplot(232)
# data3 = np.loadtxt("../Datasets_PRML_A2/Dataset_3_Team_39.csv", delimiter=',', dtype=None)
# for class_val in classes :
#     idx = (data3[:,2]==class_val)
#     plt.scatter(data3[idx,0],data3[idx,1],label="class "+str(class_val),s=10)
# plt.tight_layout()

# # plt.legend()
# # # plt.savefig("results/Dataset3_scatter")

# plt.subplot(233)
# data4 = np.loadtxt("../Datasets_PRML_A2/Dataset_4_Team_39.csv", delimiter=',', dtype=None)
# for class_val in classes :
#     idx = (data4[:,2]==class_val)
#     plt.scatter(data4[idx,0],data4[idx,1],label="class "+str(class_val),s=10)
# plt.tight_layout()

# # plt.legend()
# # plt.savefig("results/Dataset4_scatter")


# plt.subplot(234)
# data5 = np.loadtxt("../Datasets_PRML_A2/Dataset_5_Team_39.csv", delimiter=',', dtype=None)
# for class_val in classes :
#     idx = (data5[:,2]==class_val)
#     plt.scatter(data5[idx,0],data5[idx,1],label="class "+str(class_val),s=10)

# ax = plt.subplot(236)
# legend_elements = [ Line2D([0], [0], marker='o', color='b', label='class 0', markerfacecolor='blue', markersize=6, alpha=0.7),  Line2D([0], [0], marker='o', color='orange', label='class 1', markerfacecolor='orange', markersize=6)]
# ax.legend(handles=legend_elements, loc='center')
# plt.tight_layout()

# plt.show()
# plt.savefig("results/datasets_scatter")
# # plt.savefig("results/Dataset5_scatter")
