import matplotlib.pyplot as plt
import numpy as np
from sklearn import mixture

np.random.seed(1)

mleArr = []
bic = []
mleDifferences = []


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
		print np.shape(obs)
		g.fit(obs)
		mleArr.append(mle(g, obs))
		bic.append(g.bic(obs))


mleDifferences = np.diff(mleArr)


print mleDifferences
# plt.figure(1)
# plt.plot(numClusters, bic)
# plt.ylabel('BIC')
# plt.xlabel('Number of Clusters')
# plt.title('BIC plot')

plt.figure(2)
plt.plot(numClusters, mleArr)
plt.ylabel('Maximum Likelihood Estimator')
plt.xlabel('Number of Clusters')
plt.title('MLE plot')
# plt.show()

plt.figure(3)
plt.plot(mleDifferences, 'o')
plt.title('MLE Differences')
plt.ylabel('Difference in Maximum Likelihood Estimator')
plt.xlabel('Number of Clusters')
plt.show()

