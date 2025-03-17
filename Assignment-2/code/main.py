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
    parser.add_argument('-q','--question', required=True,
        choices=['1', '2.1', '2.2'])

    io_args = parser.parse_args()
    question = io_args.question


    if question == '1':
        X = utils.load_dataset('clusterData')['X']

        model = kmeans.fit(X, k=4)
        utils.plot_2dclustering(X, model['predict'](model, X))
        print("Displaying figure...")
        plt.show()
        # part 1: implement kmeans.error (which means to complete the error function in kmeans.py)
        # part 2: get clustering with lowest error out of 50 random initialization


    if question == '2.1':
        X = utils.load_dataset('clusterData2')['X']
        model = dbscan.fit(X, radius2=1, min_pts=3)
        y = model['predict'](model, X)
        utils.plot_2dclustering(X,y)
        print("Displaying figure...")
        plt.show()

    if question == '2.2':
        dataset = utils.load_dataset('animals')
        X = dataset['X']
        animals = dataset['animals']
        traits = dataset['traits']

        model = kmeans.fit(X, k=5)
        y = model['predict'](model, X)

        for kk in range(max(y)+1):
            print('Cluster {}: {}'.format(kk+1, ' '.join(animals[y==kk])))
