import numpy as np
import matplotlib.pyplot as plt
import utils


def fit(X, k, do_plot=False):
    N, D = X.shape
    y = np.ones(N)

    means = np.zeros((k, D))
    for kk in range(k):
        i = np.random.randint(N)
        means[kk] = X[i]

    while True:
        y_old = y

        # Compute euclidean distance to each mean
        dist2 = utils.euclidean_dist_squared(X, means)
        dist2[np.isnan(dist2)] = np.inf
        y = np.argmin(dist2, axis=1)

        # Update means
        for kk in range(k):
            means[kk] = X[y == kk].mean(axis=0)

        changes = np.sum(y != y_old)
        print('Running K-means, changes in cluster assignment = {}'.format(changes))

        # Stop if no point changed cluster
        if changes == 0:
            break

    model = dict()
    model['means'] = means
    model['predict'] = predict
    model['error'] = error

    if do_plot and D == 2:
        utils.plot_2dclustering(X, y)
        print("Displaying figure...")
        plt.show()

    return model


def predict(model, X):
    means = model['means']
    dist2 = utils.euclidean_dist_squared(X, means)
    dist2[np.isnan(dist2)] = np.inf
    return np.argmin(dist2, axis=1)


def error(model, X):
    # Get the cluster assignments for each point in X
    y = model['predict'](model, X)
    means = model['means']

    # Compute the squared distances to all means
    dist2 = utils.euclidean_dist_squared(X, means)

    # For each point, select the squared distance to its assigned mean
    error = np.sum(dist2[np.arange(X.shape[0]), y])

    return error
