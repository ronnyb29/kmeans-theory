import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
import sklearn.datasets


np.random.seed(1)

mleArr = np.empty([9, 9])
bic = np.empty([9, 9])
mleDifferences = np.empty([9, 8])




def compute_bic(kmeans,X):
    """
    Computes the BIC metric for a given clusters

    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn

    X     :  multidimension np array of data points

    Returns:
    -----------------------------------------
    BIC value
    """

    #number of clusters
    m = kmeans.n_clusters

    #size of data set
    N, d = X.shape

    mle = compute_mle(kmeans, X)

    const_term = 0.5 * m * np.log(N) * (d+1)

    BIC = -mle + const_term

    return(BIC)

def compute_mle(kmeans, X):
	"""
    Computes the BIC metric for a given clusters

    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn

    X     :  multidimension np array of data points

    Returns:
    -----------------------------------------
    MLE value
    """
    # assign centers and labels
	centers = [kmeans.cluster_centers_]
	labels  = kmeans.labels_
	#number of clusters
	m = kmeans.n_clusters
	# size of the clusters
	n = np.bincount(labels)
	#size of data set
	N, d = X.shape

	#compute variance for all clusters beforehand
	cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]], 'euclidean')**2) for i in range(m)])

	MLE = np.sum([n[i] * np.log(n[i]) -
	n[i] * np.log(N) -
	((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
	((n[i] - 1) * d/ 2) for i in range(m)])

	return(MLE)


numClusters = [i for i in range(1,10)]
obs = np.random.randn(100, 1)
for i in xrange(1,10):
	obs = np.concatenate((obs, i*300 + np.random.randn(100, 1)))
	for x in xrange(1,10):
		kmeans = cluster.KMeans(n_clusters = x, init='k-means++')
		kmeans.fit(obs)
		mleArr[i - 1, x - 1] = compute_mle(kmeans, obs)
		bic[i - 1, x - 1] = compute_bic(kmeans, obs)
	mleDifferences[i-1] = np.diff(mleArr[i-1])