import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# No class imbalance.
def get_plot(path_to_csv,num) :
    data = np.loadtxt(path_to_csv, delimiter=',', dtype=None)
    classes = np.array(np.unique(data[:,2], return_counts=False),dtype=int)
    for class_val in classes :
        idx = (data[:,2]==class_val)
        plt.scatter(data[idx,0],data[idx,1],label="class "+str(class_val),s=10)
        plt.title("Dataset "+str(num),size=10)
        plt.tight_layout()
    plt.legend(loc="lower right",prop={'size': 6})

get_plot("../Datasets_PRML_A2/Dataset_1_Team_39.csv",1)
plt.savefig("results/Dataset"+str(1)+"_scatter")

fig = plt.figure()

for i in range(2,6) :
    plt.subplot(219+i)
    get_plot("../Datasets_PRML_A2/Dataset_"+str(i)+"_Team_39.csv",i)

plt.savefig("results/Dataset2345_scatter")
