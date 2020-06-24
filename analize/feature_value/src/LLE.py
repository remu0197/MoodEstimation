import numpy as np #numpyという行列などを扱うライブラリを利用
import pandas as pd #pandasというデータ分析ライブラリを利用
import matplotlib.pyplot as plt #プロット用のライブラリを利用
from sklearn import preprocessing, decomposition, manifold, manifold #機械学習用のライブラリを利用
from sklearn import datasets #使用するデータ
import csv
from mpl_toolkits.mplot3d import Axes3D

path = '../data/half/friendly.csv'
no_label_path = '../data/half/no_label.csv'

labels = []
data = []
no_label_count = 1

# with open(path) as f :
#     reader = csv.reader(f)
#     header = next(reader)
#     for row in reader :
#         no_label_count += 1
#         labels.append(float(row[2])/5.0)
#         data.append(row[3:])

# with open(no_label_path) as f :
#     reader = csv.reader(f)
#     header = next(reader)
#     for row in reader :
#         data.append(row[3:])

Y = np.array(labels)
X = np.array(data)
# 2：moon型のデータを読み込む--------------------------------
# X,Y = datasets.make_moons(n_samples=200, noise=0.05, random_state=0)
 
# 3：データの整形-------------------------------------------------------
# sc=preprocessing.StandardScaler()
# sc.fit(X)
# X_norm=sc.transform(X)
 
# 4：Isomapを実施-------------------------------
# isomap = manifold.Isomap(n_neighbors=10, n_components=2)
# X_isomap = isomap.fit_transform(X)
 
# 解説5：LLEを実施-------------------------------
lle = manifold.LocallyLinearEmbedding(n_neighbors=25, n_components=4, method='modified')
# lle = manifold.TSNE(n_components=5, init='pca', random_state=0, method='exact')
X_lle = lle.fit_transform(X)
 
 
# 6: 結果をプロットする-----------------------------
# matplotlib inline
 
# plt.figure(figsize=(10,10))
# plt.subplot(3, 1, 1)
# plt.scatter(X[:,0],X[:,1], c=Y)
# plt.xlabel('x')
# plt.ylabel('y')
 
# plt.subplot(211)
# plt.scatter(X_isomap[:,0],X_isomap[:,1], c=Y)
# plt.xlabel('IM-1')
# plt.ylabel('IM-2')
 
# plt.scatter(X_lle[no_label_count+1:,1], X_lle[no_label_count+1:,0], facecolor='None',edgecolor='red') 
# plt.scatter(X_lle[1:no_label_count,1], X_lle[1:no_label_count,0])
# plt.xlabel('LLE-1')
# plt.ylabel('LLE-2')
 
# plt.show

# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(X_lle[1:no_label_count,0], X_lle[1:no_label_count,1], X_lle[1:no_label_count,2], c=Y)
# ax.scatter(X_lle[no_label_count+1:,0], X_lle[no_label_count+1:,1], X_lle[no_label_count+1:,2], facecolor='None',edgecolor='orange', alpha=0.1)


# plt.show()