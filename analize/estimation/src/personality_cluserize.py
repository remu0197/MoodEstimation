from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt 
import random
import sys
import csv
import copy

def personality_clusterizing(t1, t2) :
    result = [t1, t2]
    df = pd.read_csv('../data/big5_results_min.csv')
    data = np.array([df[t1].tolist(), df[t2].tolist()], np.float32)

    data = data.T
    data_n = np.linalg.norm(data,ord=2)
    data_norm = data / data_n
    a = random.randrange(200)

    x,y = data_norm[:,0], data_norm[:,1]
    c = KMeans(n_clusters=2, init='k-means++', n_init=10000, max_iter=100000,
                   tol=0.001,precompute_distances='auto', verbose=0,
                   random_state=a, copy_x=True, n_jobs=1)
    pred=c.fit(data_norm)

    plt.scatter(x,y,c=pred.labels_)
    plt.savefig('./clustering/results/fig/' + t1 + '_' + t2 + '.png')

    result.extend(pred.labels_)

    return result

if __name__ == "__main__":
    # traits = ['an', 'in', 'un', 'en', 'on']
    # results = []
    # for i, t1 in enumerate(traits) :
    #     temp = copy.copy(traits)
    #     temp.pop(i)
    #     for t2 in temp :
    #         print(t1+t2)
    #         results.append(personality_clusterizing(t1, t2))

    results = personality_clusterizing('en', 'on')

    with open('./clustering/results/temp.csv', 'w') as f :
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(results)