import sys
from sklearn import mixture
import matplotlib
matplotlib.use('Agg') # -----(1)
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

class Clusterizer :
    def __init__(self, data) :
        self.__data = data
        self.__labels = [0 for _ in range(len(data))]
        self.__x, self.__o = [], []

    # VBGMMで実装
    def predict(self, method="",should_remove_filler=False) :
        if method == "VBGMM" :
            # n_component: クラスタの最大数
            vbgm = mixture.BayesianGaussianMixture(
                n_components=3,
                random_state=456
            )

            vbgm.fit(self.__data)
            self.__labels = vbgm.predict(self.__data)
            self.__label_count = 0
            labels = []
            for label in self.__labels :
                if not label in labels :
                    labels.append(label)
                    self.__label_count += 1
        elif method == "SMIRNOV" :
            # alpha: 有意水準
            alpha = 0.05
            n = len(self.__data)
            temp = self.__data

            while n > 0:
                t = stats.t.isf(q=((alpha/n)/2), df=(n - 2))
                tau = (n - 1) * t / np.sqrt(n * (n - 2) + n * t * t)
                min_interval, max_interval = 40, -1
                min_index, max_index = 0, 0
                for i, x in enumerate(self.__data) :
                    if self.__labels[i] == 1 :
                        continue
                    if min_interval > x :
                        min_interval = x
                        min_index = i
                    if max_interval < x :
                        max_interval = x
                        max_index = i
                
                myu, std = np.mean(temp), np.std(temp, ddof=1)
                if np.abs(max_interval - myu) > np.abs(min_interval - myu) :
                    if np.abs((self.__data[max_index] - myu) / std) < tau :
                        break
                    self.__labels[max_index] = 1
                    temp.remove(max_interval)
                else :
                    if np.abs((self.__data[min_index] - myu) / std) < tau :
                        break
                    self.__labels[min_index] = 1
                    temp.remove(min_interval)

                n = n - 1 
        else :
            print("ERROR")
            sys.exit()

        return self.__labels

    def plot(self) :
        x, y = [], []
        for temp in self.__data :
            x.append(temp[0])
            y.append(temp[1])

        plt.scatter(x,y,c=self.__labels)
        plt.show()

    def save_plot(self, path) :
        x, y, labels = [], [], []
        for value in self.__x :
            x.append(value)
            y.append(0)
            labels.append(0)

        for value in self.__o :
            x.append(value)
            y.append(0)
            labels.append(1)

        plt.scatter(x,y,c=labels)

        plt.savefig(path)

    def get_labels(self):
        return self.__labels

    def get_middle_data(self) :
        middle_label = 0

        value, label = [], []
        for i, x in enumerate(self.__data) :
            if not self.__labels[i] in label :
                value.append(x[0])
                label.append(self.__labels[i])

        print(self.__labels)

        if self.__label_count == 1 :
            middle_label = label[0]
        elif self.__label_count == 2 :
            if value[0] > value[1] :
                middle_label = label[1]
            else :
                middle_label = label[0]
        else :
            if value[0] > value[1] and value[1] > value[2] or value[0] < value[1] and value[1] < value[2] :
                middle_label = label[1]
            elif value[1] > value[2] and value[2] > value[0] or value[1] < value[2] and value[2] < value[0] :
                middle_label = label[2]
            else :
                middle_label = label[0]

        result = []
        for i, x in enumerate(self.__data) :
            if self.__labels[i] == middle_label :
                result.append(x[0])

        return result

    # def get_clusterized_data(self) :
    #     data = [[], []]
    #     for i, label in self.__labels :

if __name__ == "__main__":
    cluster = Clusterizer([])
    cluster.predict("VBGMM")