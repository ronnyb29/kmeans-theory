import matplotlib.pyplot as plt
import numpy as np
from sklearn import mixture

np.random.seed(1)

mleArr = np.empty([2, 9])
bic = np.empty([2, 9])
mleDifferences = np.empty([2, 8])


def verifyConcavity(arr):
	for x in xrange(0, len(arr) - 2):
		if arr[x+2] - 2*arr[x +1] + arr[x] >  0:
			return False
	return True

def mle(GMM, X):
	return 2 * GMM.score(X).sum()
			

numClusters = [i for i in range(1,10)]
obs = np.random.randn(100, 1)
for i in [1, 2]:
	obs = np.concatenate((obs, i*100 + np.random.randn(100, 1)))
	for x in xrange(1,10):
		g = mixture.GMM(n_components=x)
		# generate random observations with two modes centered on 0 and 100 
		g.fit(obs)
		mleArr[i - 1, x - 1] = mle(g, obs)
		bic[i - 1, x - 1] = g.bic(obs)
	mleDifferences[i-1] = np.diff(mleArr[i-1])


# 
# print mleDifferences

#plotting a bunch of stuff
plt.figure(1)
plt.plot(numClusters, bic[0])
plt.ylabel('BIC')
plt.xlabel('Number of Clusters')
plt.title('BIC plot')
plt.show()

plt.figure(2)
plt.plot(numClusters, mleArr[0])
plt.ylabel('Maximum Likelihood Estimator')
plt.xlabel('Number of Clusters')
plt.title('MLE plot')
plt.show()

print 'shape of numClusters is ' +  np.shape(numClusters[0:-2])
print 'shape of mleDifferences[0] is ' + np.shape(mleDifferences[0])

plt.figure(3)
plt.plot(numClusters[0:-2], mleDifferences[0], 'o')
plt.title('MLE Differences')
plt.ylabel('Difference in Maximum Likelihood Estimator')
plt.xlabel('Number of Clusters')
plt.show()

