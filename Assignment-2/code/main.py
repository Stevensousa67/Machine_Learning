import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np

import utils
import kmeans
import dbscan

if __name__ == '__main__':
    argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--question', required=True,
                        choices=['1', '2.1', '2.2'])

    io_args = parser.parse_args()
    question = io_args.question

    if question == '1':
        X = utils.load_dataset('clusterData')['X']

        # Parameters for kmeans
        k = 4
        num_runs = 50

        # Store the best model here
        best_model = None
        best_error = np.inf

        for run in range(num_runs):
            print(f"Run {run+1}/{num_runs}...")
            model = kmeans.fit(X, k, do_plot=False)
            error = model['error'](model, X)

            if error < best_error:
                best_model = model
                best_error = error

        y_best = best_model['predict'](best_model, X)
        utils.plot_2dclustering(X, y_best)
        print(f"Best error: {best_error}")
        print("Displaying the best clustering...")
        plt.show()

    if question == '2.1':
        X = utils.load_dataset('clusterData2')['X']
        model = dbscan.fit(X, radius2=4, min_pts=3)
        y = model['predict'](model, X)
        utils.plot_2dclustering(X, y)
        print("Displaying figure...")
        plt.show()

    if question == '2.2':
        dataset = utils.load_dataset('animals')
        X = dataset['X']
        animals = dataset['animals']
        traits = dataset['traits']

        # dbscan model
        model = dbscan.fit(X, radius2=15, min_pts=3)
        y = model['predict'](model, X)
        unique_clusters = np.unique(y[y != -1])
        print(f"Number of clusters: {len(unique_clusters)}")
        for cluster in unique_clusters:
            if cluster != -1:
                print(f"Cluster {cluster+1}: {', '.join(animals[y == cluster])}")
        if -1 in y:
            print(f"Noise: {', '.join(animals[y == -1])}")

        # kmeans model
        # model = kmeans.fit(X, k=5)
        # y = model['predict'](model, X)

        # for kk in range(max(y)+1):
        #     print('Cluster {}: {}'.format(kk+1, ' '.join(animals[y == kk])))
