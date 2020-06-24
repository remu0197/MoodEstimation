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

def set_color(l):
    if l == 0:
        return "b"  # blue
    elif l == 1:
        return "g"  # green
    else:
        return "r"  # red

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

        color_list = map(set_color, pred.labels_)

        print(pred.labels_)
        plt.scatter(x,y,c=pred.labels_,cmap='RdYlGn')
        plt.savefig('./clustering/results/fig/' + t1 + '_' + t2 + '.png')

        result.extend(pred.labels_)

        return result

if __name__ == "__main__":
        traits = ['an', 'on']
        # results = []
        # for i, t1 in enumerate(traits) :
        #         temp = traits.pop(i)
        #         for t2 in temp :
        #                 print(t1+t2)
        #                 results.append(personality_clusterizing(t1, t2))

        # with open('./results/personality.csv', 'w') as f :
        #         writer = csv.writer(f, lineterminator='\n')
        #         writer.writerow(results)

        personality_clusterizing('an', 'on')


# a = random.randrange(200)
# # c = KMeans(n_clusters=3, init='k-means++', n_init=10000, max_iter=100000,
# #                        tol=0.001,precompute_distances='auto', verbose=0,
# #                        random_state=a, copy_x=True, n_jobs=1)
# x=X_norm[:,0]
# y=X_norm[:,1]

# c = BayesianGaussianMixture(n_components=3,random_state=a)
# print(a)
# pred = c.fit(data)
# plt.subplot(4, 1, 3)
# plt.scatter(x,y, c=labels)
 
# # 解説9: クラスター数の確率結果をプロット-----------------------------------------------
# #print(vbgm.weights_)
# plt.subplot(4, 1, 4)
# x_tick =np.array([1,2,3,4,5,6,7,8,9,10])
# plt.bar(x_tick, vbgm.weights_, width=0.7, tick_label=x_tick)
# plt.show()

# sys.exit()

# for i in range(5) :
#     labels = data[pred == i]
#     plt.scatter(labels[:, 0], labels[:, 1])

# centers = c.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], s = 100, facecolors='none', edgecolors='black')

# print('Distortion: %.2f'% c.inertia_)
# plt.show()

# from sklearn.metrics import silhouette_samples
# from matplotlib import cm

# cluster_labels = np.unique(pred)       # y_kmの要素の中で重複を無くす
# n_clusters=cluster_labels.shape[0]     # 配列の長さを返す。つまりここでは n_clustersで指定した3となる

# # シルエット係数を計算
# silhouette_vals = silhouette_samples(data,pred,metric='euclidean')  # サンプルデータ, クラスター番号、ユークリッド距離でシルエット係数計算
# y_ax_lower, y_ax_upper= 0,0
# yticks = []

# for i,c in enumerate(cluster_labels):
#         c_silhouette_vals = silhouette_vals[pred==c]      # cluster_labelsには 0,1,2が入っている（enumerateなのでiにも0,1,2が入ってる（たまたま））
#         c_silhouette_vals.sort()
#         y_ax_upper += len(c_silhouette_vals)              # サンプルの個数をクラスターごとに足し上げてy軸の最大値を決定
#         color = cm.jet(float(i)/n_clusters)               # 色の値を作る
#         plt.barh(range(y_ax_lower,y_ax_upper),            # 水平の棒グラフのを描画（底辺の範囲を指定）
#                          c_silhouette_vals,               # 棒の幅（1サンプルを表す）
#                          height=1.0,                      # 棒の高さ
#                          edgecolor='none',                # 棒の端の色
#                          color=color)                     # 棒の色
#         yticks.append((y_ax_lower+y_ax_upper)/2)          # クラスタラベルの表示位置を追加
#         y_ax_lower += len(c_silhouette_vals)              # 底辺の値に棒の幅を追加

# silhouette_avg = np.mean(silhouette_vals)                 # シルエット係数の平均値
# plt.axvline(silhouette_avg,color="red",linestyle="--")    # 係数の平均値に破線を引く 
# plt.yticks(yticks,cluster_labels + 1)                     # クラスタレベルを表示
# plt.ylabel('Cluster')
# plt.xlabel('silhouette coefficient')
# plt.show()