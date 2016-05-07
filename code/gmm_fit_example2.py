import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

# Number of samples per component
n_samples = 100

# Generate random sample, two components
np.random.seed(0)
X = np.r_[np.random.randn(100, 2),
          np.random.randn(n_samples, 2) + np.array([300, 0]),
          np.random.randn(n_samples, 2) + np.array([600, 0])]

# Fit a mixture of Gaussians with EM using five components
gmm = mixture.GMM(n_components=3, covariance_type='full')
gmm.fit(X)


color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])

for i, (clf, title) in enumerate([(gmm, 'GMM clustered into 3 components')]):
    splot = plt.plot()
    Y_ = clf.predict(X)
    for i, (mean, covar, color) in enumerate(zip(
            clf.means_, clf._get_covars(), color_iter)):
        v, w = linalg.eigh(covar)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
        ell.set_alpha(0.5)

    plt.xlim(-5, 605)
    plt.ylim(-1, 1)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)

plt.show()