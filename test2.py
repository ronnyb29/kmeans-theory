import matplotlib.pyplot as plt
import numpy as np
from sklearn import mixture

np.random.seed(1)

mleArr = []
bic = []
mleDifferences = []


def mle(GMM, X):
	return 2 * GMM.score(X).sum()

def verifyConcavity(arr):
	for x in xrange(0, len(arr) - 2):
		if arr[x+2] - 2*arr[x +1] + arr[x] >  0:
			return False
	return True
			
numClusters = [i for i in range(1,10)]
for x in xrange(1,10):
	g = mixture.GMM(n_components=x)
	# generate random observations with two modes centered on 0 and 100 
	obs = np.concatenate((np.random.randn(100, 1), 100 + np.random.randn(300, 1)))
	g.fit(obs)
	print 'mle'
	print mle(g, obs)
	mleArr.append(mle(g, obs))
	bic.append(g.bic(obs))
