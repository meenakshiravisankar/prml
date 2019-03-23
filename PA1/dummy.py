import numpy as np
import matplotlib.pyplot as plt
mini = np.array([-600,-143])
maxi = np.array([671,56.9])

xy = np.mgrid[mini[0]:maxi[0]:2, mini[1]:maxi[1]:2].reshape(2,-1)

class1x,class1y = [],[]
class2x,class2y = [],[]
class3x,class3y = [],[]

X_data = np.transpose(xy)
y_data = np.zeros(X_data.shape)
print(X_data.shape)

for i in range(xy.shape[1]) :
    for j in range(xy.shape[1]):
        
        
        print(xy[0,:])

        raise SystemExit
		#print(A[3].shape)
		# if(np.argmax(A[3],axis=0)==0):
		# 	class1x.append(xy[0,j])
		# 	class1y.append(xy[1,j])
		# elif(np.argmax(A[3],axis=0)==1):
		# 	class2x.append(xy[0,j])
		# 	class2y.append(xy[1,j])
		# else:
		# 	class3x.append(xy[0,j])
		# 	class3y.append(xy[1,j])

# plt.scatter(trainX[0][np.squeeze(np.where(trainY==0))],trainX[1][np.squeeze(np.where(trainY==0))], color='darkred')
# plt.scatter(trainX[0][np.squeeze(np.where(trainY==1))],trainX[1][np.squeeze(np.where(trainY==1))], color='darkgreen')
# plt.scatter(trainX[0][np.squeeze(np.where(trainY==2))],trainX[1][np.squeeze(np.where(trainY==2))], color='navy')
# plt.scatter(class1x, class1y, color='orangered', alpha=0.01)
# plt.scatter(class2x, class2y, color='lawngreen', alpha=0.01)
# plt.scatter(class3x, class3y, color='deepskyblue', alpha=0.01)
plt.title("Decision boundaries")
plt.show()