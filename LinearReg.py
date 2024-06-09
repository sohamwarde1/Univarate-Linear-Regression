import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([15,14,13,12,11,10])
y_train = np.array([741,583,446,435,359,330])


def costf(x_train,y_train,w,b):

	err = 0
	m  = len(x_train)

	for i in range(n):
		f = (w*x_train[i])+b
		err = err + ((f-y_train[i])**2)

	cost = err/(2*m)

	return cost


def compute_gradient(x_train,y_train,w,b):
	 
	 wv = 0
	 bv = 0
	 m = x.shape[0]

	 for i in range(m):
	 	f = (w*x_train[i])+b

	 	wv = wv + ((f-y_train[i])*x_train[i])

	 	bv = bv + ((f-y_train[i]))


	 return wv/m , bv/m


def gradientdescent(wi, bi, x_train, y_train, alpha):

	minc = costf(x_train,y_train,wi,bi)
	wlist = [wi]
	blist = [bi]

	while True:
		wn1,bn1 = compute_gradient(x_train, y_train, wlist[-1], blistp-1)

		wnew = wlist[-1] - (alpha*wn1)
		bnew = blist[-1] - (alpha*bn1)

		cost = costf(x_train,y_train,w,b)

		if cost<=minc:
			minc = cost
		else:
			return wlist[-1],blist[-1]

		wlist.append(wnew)
		blist.append(bnew)

		
def plotline(w,b,x_train):
	wi = 0
	bi = 0
	



plt.scatter(x_train,y_train,c = 'r')
plt.show()