import matplotlib.pyplot as plt
import numpy as np
from sklearn import mixture
from sklearn import cluster

np.random.seed(1)

mleArr = np.empty([10, 10])
bic = np.empty([10, 10])
mleDifferences = np.empty([10, 9])
mleProperties = np.empty([10, 9])


def verifyConcavity(arr):
	for x in xrange(0, len(arr) - 2):
		if arr[x+2] - 2*arr[x +1] + arr[x] >  0:
			return False
	return True

def mle(GMM, X):
	return GMM.score(X).sum() #score computes the log probability under the model, MLE by def is just the sum (assumes log likelihood)
			

numClusters = range(1, 11)
obs = np.random.randn(100, 1)
for i in xrange(1,11):
	obs = np.concatenate((obs, i*300 + np.random.randn(100, 1)))
	for x in xrange(1,11):
		kmeans_init = cluster.KMeans(n_clusters = x, init='k-means++')
		kmeans_init.fit(obs)
		g = mixture.GMM(covariance_type='tied',  n_components=x, n_init=1, n_iter=100,  random_state=1, init_params = 'wc')
		g.means_ = kmeans_init.cluster_centers_
		# generate random observations with two modes centered on 0 and 100 
		#n init makes it very slow
		g.fit(obs)
		mleArr[i - 1, x - 1] = mle(g, obs)

		bic[i - 1, x - 1] = g.bic(obs)
	mleDifferences[i-1] = np.diff(mleArr[i-1])
	mleProperties[i-1] = mleDifferences[i-1] - np.log(obs.shape[0]) #subtract difference in penalties


#contruct probabiltiy distribution
probabiltiyCorrect = np.empty([10, 2])
for i in xrange(0,10):
	#go from 0 to 9. indexing via MLE differences. correct number is k-1, so actually i+1
	opt_k = np.argmin(bic[i]) #gets optimal k for BIC for sample from i+2 clusters
	theoretical_gt = opt_k
	theoretical_lt = 9 - theoretical_gt
	actual__gt  = 0 #actual number of correct greater thans
	actual__lt  = 0 #actual number of correct greater thans
	for x in xrange(0, 9):
		if mleProperties[i, x] >= 0 and x < theoretical_gt:
			actual__gt += 1
		if mleProperties[i, x] <= 0 and x >= theoretical_gt:
			actual__lt += 1
	percent_gt = actual__gt / theoretical_gt
	if theoretical_lt > 0:
		percent_lt = actual__lt / theoretical_lt
	else:
		percent_lt = 1
	
	probabiltiyCorrect[i] = [percent_gt, percent_lt]

#bic[0] corresponds to 2 'real' cluster centers
#7 is the strange case

#plotting a bunch of stuff
plt.figure(1)
plt.plot(numClusters, bic[1])
plt.ylabel('BIC')
plt.xlabel('Number of Clusters')
plt.title('BIC plot')
plt.show()

plt.figure(2)
plt.plot(numClusters, mleArr[1])
plt.ylabel('Maximum Likelihood Estimator')
plt.xlabel('Number of Clusters')
plt.title('MLE plot')
plt.show()

plt.figure(3)
plt.plot(numClusters[0:-1], mleDifferences[1], 'o')
plt.title('MLE Differences')
plt.ylabel('Difference in Maximum Likelihood Estimator')
plt.xlabel('Number of Clusters')
# plt.show()

plt.figure(4)
plt.plot(numClusters[0:-1], mleProperties[1], 'o')
plt.title('Necessary Condition on MLE')
plt.ylabel('Difference in Maximum Likelihood Estimator')
plt.xlabel('Number of Clusters')
plt.show()

