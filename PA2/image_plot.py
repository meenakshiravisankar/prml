import numpy as np
from matplotlib import pyplot as plt

img_array = np.load('../Datasets_PRML_A2/Image_Dataset/fan.npy')
plt.imshow(img_array, cmap='gray')
plt.savefig("results/fan.png")
