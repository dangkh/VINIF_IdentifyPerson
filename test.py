import numpy as np


def conv2(features, filterSize):
	ft = features.T
	output = np.mean(np.log([np.convolve(ft[x], np.ones(filterSize), 'valid') / filterSize for x in range(ft.shape[0])]), axis = 0)
	return  output


a = [np.random.randint(1,100) for ii in range(10)]
print(a)
print(np.convolve(a, np.ones(3), 'valid') / 3)