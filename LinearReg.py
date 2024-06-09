import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([15,14,13,12,11,10])
y_train = np.array([741,583,446,435,359,330])

plt.scatter(x_train,y_train,c = 'r')
plt.show()

def costf(x_train,y_train,w,b):

	err = 0
	m  = len(x_train)

	for i in range(n):
		f = (w*x_train[i])+b
		err = err + ((f-y_train[i])**2)

	cost = err/(2*m)

	return cost


