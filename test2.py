import matplotlib.pyplot as plt
import numpy as np
from sklearn import mixture

np.random.seed(1)

mleArr = np.empty([9, 9])
bic = np.empty([9, 9])
mleDifferences = np.empty([9, 8])


def verifyConcavity(arr):
	for x in xrange(0, len(arr) - 2):
		if arr[x+2] - 2*arr[x +1] + arr[x] >  0:
			return False
	return True

def mle(GMM, X):
	return GMM.score(X).sum() #score computes the log probability under the model., MLE is just the sum (assumes log likelihood)
			

numClusters = [i for i in range(1,10)]
obs = np.random.randn(100, 1)
for i in xrange(1,10):
	obs = np.concatenate((obs, i*100 + np.random.randn(100, 1)))
	for x in xrange(1,10):
		g = mixture.GMM(n_components=x, n_init=1, n_iter=100,  random_state=1)
		# generate random observations with two modes centered on 0 and 100 
		#n init makes it very slow
		g.fit(obs)
		mleArr[i - 1, x - 1] = mle(g, obs)

		bic[i - 1, x - 1] = g.bic(obs)
	mleDifferences[i-1] = np.diff(mleArr[i-1])


#bic[0] corresponds to 2 'real' cluster centers

#plotting a bunch of stuff
plt.figure(1)
plt.plot(numClusters, bic[6])
plt.ylabel('BIC')
plt.xlabel('Number of Clusters')
plt.title('BIC plot')
plt.show()

plt.figure(2)
plt.plot(numClusters, mleArr[6])
plt.ylabel('Maximum Likelihood Estimator')
plt.xlabel('Number of Clusters')
plt.title('MLE plot')
plt.show()

plt.figure(3)
plt.plot(numClusters[0:-1], mleDifferences[6], 'o')
plt.title('MLE Differences')
plt.ylabel('Difference in Maximum Likelihood Estimator')
plt.xlabel('Number of Clusters')
plt.show()

