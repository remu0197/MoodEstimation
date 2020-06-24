import pandas as pd
from sklearn import preprocessing, cluster, mixture
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import LinearSegmentedColormap

def generate_cmap(colors):
    """自分で定義したカラーマップを返す"""
    values = range(len(colors))

    vmax = np.ceil(np.max(values))
    color_list = []
    for v, c in zip(values, colors):
        color_list.append( ( v/ vmax, c) )
    return LinearSegmentedColormap.from_list('custom_cmap', color_list)

# read big5 data
df = pd.read_csv('../data/big5_results_min.csv')

# processing data
X = df[["an", "on"]]
print(len(X))
sc = preprocessing.StandardScaler()
sc.fit(X)
X_norm = sc.transform(X)
# X_norm = np.array([df['an'].tolist(), df['on'].tolist()], np.float32).T

x, y = X_norm[:,0], X_norm[:,1]
# plt.figure(figsize=(10,10))
# plt.subplot(4,1,1)
# plt.scatter(x,y)

def set_color(l):
    if l == 0:
        return "b"  # blue
    elif l == 1:
        return "r"  # green
    else:
        return "g"  # red

km = cluster.KMeans(n_clusters=2, init='k-means++', random_state=100)
z_km = km.fit(X_norm)

cm = generate_cmap(['#87CEEB', '#2E8B57', '#F4A460'])

# plt.subplot(4,1,2)
plt.scatter(x,y,c=z_km.labels_, cmap=cm)
print(z_km.labels_)
df_label = df['ID']
df_label['LABEL'] = z_km.labels_
df_label.to_csv('./label.csv')
# plt.scatter(z_km.cluster_centers_[:,0],z_km.cluster_centers_[:,1],s=250, marker='*',c='red')
plt.show()

vbgmm = mixture.BayesianGaussianMixture(n_components=10, random_state=6)
z_vbgmm = vbgmm.fit(X_norm)

plt.subplot(4,1,3)
plt.scatter(x, y, c=z_vbgmm.predict(X_norm))

plt.subplot(4,1,4)
x_tick = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
plt.bar(x_tick, z_vbgmm.weights_, width=0.7,tick_label=x_tick)
plt.show()
