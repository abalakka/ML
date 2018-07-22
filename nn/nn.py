import numpy as np


def sigmoid(x):
	return 1/(1+np.exp(-x))

def der_sigmoid(x):
	return sigmoid(x)*(1-sigmoid(x))

x = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1] ])
#x = x.reshape(4,-1)
y = np.array([[0, 0, 1, 1]]).T

np.random.seed(1)

w1 = np.random.random((3,1)) - 1

for _ in range(4000):
	l0 = x
	l1 = sigmoid(np.dot(x, w1))

	err = y - l1

	corr = err*der_sigmoid(l1)

	w1 += np.dot(l0.T, corr)


print(l1)
