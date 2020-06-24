import sys
from sklearn import mixture

class Clusterizer :
    def __init__(self, data) :
        # self.__data = data
        self.__data = [
            [0,0], [0,0], [0,0], [0,0],
            [1,1], [1,1], [1,1], [1,1],
            [2,2], [2,2], [2,2], [2,2]
        ]

    # VBGMMで実装
    def predict(self, method) :
        if method == "VBGMM" :
            # n_component: クラスタの最大数
            vbgm = mixture.BayesianGaussianMixture(
                n_components=3, 
                random_state=456
            )

            vbgm.fit(self.__data)
            self.__labels = vbgm.predict(self.__data)
            self.__max_label = 0
            print(self.__labels)
        else :
            print("ERROR")
            sys.exit()

    def get_labels(self):
        return self.__labels

    # def get_clusterized_data(self) :
    #     data = [[], []]
    #     for i, label in self.__labels :

if __name__ == "__main__":
    cluster = Clusterizer([])
    cluster.predict("VBGMM")