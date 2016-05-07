import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
import sklearn.datasets
import matplotlib.patches as mpatches



np.random.seed(1)

mleArr = np.empty([10, 10])
bic = np.empty([10, 10])
mleDifferences = np.empty([10, 9])
mleProperties = np.empty([10, 9])


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

numClusters = [i for i in range(1,11)] #goes from 1 to 9
obs = np.random.randn(100, 1)
for i in xrange(1,11):
    obs = np.concatenate((obs, i*300 + np.random.randn(100, 1)))
    for x in xrange(1,11):
        kmeans = cluster.KMeans(n_clusters = x, init='k-means++')
        kmeans.fit(obs)
        #the guy who's stack overflow i got the inspiration from for fixing his BIC has the most pythonic way of doing these few lines, it's beautiful
        mleArr[i - 1, x - 1] = compute_mle(kmeans, obs)
        bic[i - 1, x - 1] = compute_bic(kmeans, obs)

    mleDifferences[i-1] = np.diff(mleArr[i-1])
    mleProperties[i-1] = mleDifferences[i-1] - np.log(obs.shape[0]) #subtract difference in penalties

#contruct probabiltiy distribution
probabilityCorrect = np.empty([10, 2])
for i in xrange(0,10):
    #go from 0 to 9. indexing via MLE differences. correct number is k-1, so actually i+1
    opt_k = np.argmin(bic[i]) #gets optimal k for BIC for sample from i+2 clusters. really this is opt_k - 1
    theoretical_gt = float(opt_k)
    theoretical_lt = float(9 - theoretical_gt)
    actual_gt  = float(0) #actual number of correct greater thans
    actual_lt  = float(0) #actual number of correct greater thans
    #float necessary to do proper division
    for x in xrange(0, 9):
        if mleProperties[i, x] >= 0 and x < theoretical_gt:
            actual_gt += 1
        if mleProperties[i, x] <= 0 and x >= theoretical_gt:
            actual_lt += 1
    percent_gt = actual_gt / theoretical_gt
    if theoretical_lt > 0:
        percent_lt = actual_lt / theoretical_lt
    else:
        percent_lt = 1
    
    probabilityCorrect[i] = [percent_gt, percent_lt]

#bic[0] corresponds to 2 'real' cluster centers
#7 is the strange case


#plotting a bunch of stuff
plt.figure(1)
plt.plot(numClusters, bic[7])
plt.ylabel('BIC')
plt.xlabel('Number of Clusters')
plt.title('BIC plot')
plt.show()

plt.figure(2)
plt.plot(numClusters, mleArr[7])
plt.ylabel('Maximum Likelihood Estimator')
plt.xlabel('Number of Clusters')
plt.title('MLE plot')
plt.show()

plt.figure(3)
plt.plot(numClusters[0:-1], mleDifferences[7], 'o')
plt.title('MLE Differences')
plt.ylabel('Difference in Maximum Likelihood Estimator')
plt.xlabel('Number of Clusters')
plt.show()

plt.figure(4)
plt.plot(numClusters[0:-1], mleProperties[7], 'o')
plt.title('Necessary Condition on MLE')
plt.ylabel('Difference in Maximum Likelihood Estimator')
plt.xlabel('Number of Clusters')
plt.show()


#Percentage correct
plt.figure(5)
plt.ylim(0, 1.5)
plt.plot(numClusters, probabilityCorrect[:, 0], 'g', numClusters, probabilityCorrect[:, 1], 'b')
green_patch = mpatches.Patch(color='green', label='k < k*')
blue_patch = mpatches.Patch(color='blue', label='k > k*')
plt.legend(handles=[green_patch, blue_patch])
plt.title('Probability of Necessary Condition for unimodality occuring')
plt.ylabel('probability')
plt.xlabel('Number of Clusters')
plt.show()
