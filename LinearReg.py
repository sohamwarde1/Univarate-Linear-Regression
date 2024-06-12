import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0, 2.0, 3.0, 4.0])
y_train = np.array([300.0, 500.0,800.0, 850.0])


def costf(x_train,y_train,w,b):

	err = 0
	m  = len(x_train)

	for i in range(m):
		f = (w*x_train[i])+b
		err = err + ((f-y_train[i])**2)

	cost = err/(2*m)

	return cost


def compute_gradient(x_train,y_train,w,b):
	 
	 wv = 0
	 bv = 0
	 m = x_train.shape[0]

	 for i in range(m):
	 	f = (w*x_train[i])+b

	 	wv = wv + ((f-y_train[i])*x_train[i])

	 	bv = bv + ((f-y_train[i]))


	 return wv/m , bv/m


def gradientdescent(wi, bi, x_train, y_train, alpha, iters):

	minc = costf(x_train,y_train,wi,bi)
	wlist = [wi]
	blist = [bi]
	j = []

	for i in range(iters):
		wn1,bn1 = compute_gradient(x_train, y_train, wlist[-1], blist[-1])

		wnew = wlist[-1] - (alpha*wn1)
		bnew = blist[-1] - (alpha*bn1)

		cost = costf(x_train,y_train,wnew,bnew)
		j.append(cost)


		wlist.append(wnew)
		blist.append(bnew)
	return wlist[-1],blist[-1],j


		
def createline(x_train,y_train, alpha,iters):
	wi = 0
	bi = 0
	j = []
	m = x_train.shape[0]
	
	w,b,j = gradientdescent(wi,bi,x_train,y_train,alpha,iters)
	print(w)

	f = []

	for i in range(m):
		f.append((w*x_train[i])+b)

	return f,j





plt.scatter(x_train,y_train,c = 'r')
alpha = 0.1
iters = 400
f,j = createline(x_train, y_train, alpha,iters)

plt.plot(x_train, f, c = 'b')

plt.show()

iters = np.arange(iters)

plt.plot(iters,j, c = 'b')
plt.show()