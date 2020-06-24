import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import csv

# warning 消すコード
import warnings
warnings.filterwarnings('ignore')

class SVM :
    def __init__(self, path, selects, is_minimize=True, drop_path='') :
        self.__df = pd.read_csv(path)
        if is_minimize is True :
            self.__df['ANSWER'] = self.__df['ANSWER'].apply(lambda x: 0 if int(x) <= 3 else (2 if int(x) > 3 else 1))
        self.__df = self.__df.drop(['ID', 'SECTION'],axis=1)

        if len(selects) is not 0 :
            self.__df = self.__df.loc[:, selects]
            # print(self.__df)

        if drop_path is not '' :
            f = open(drop_path, 'r')
            reader = csv.reader(f)

            for row in reader :
                for r in row :
                    self.__df = self.__df.drop(str(r), axis=1)

        self.__results = [[0] * 3 for i in range(3)]

    def predict(self, show) :
        all_X = self.__df.drop('ANSWER', axis=1)
        all_Y = self.__df.ANSWER

        # 標準化
        # sc = StandardScaler()
        # sc.fit(all_X)
        # all_X = sc.transform(all_X)

        data_list = []

        for i in range(int(len(self.__df) / 4)) :
            index = i * 4
            data = []

            train_X = all_X.drop([index, index+1, index+2, index+3])
            # sc.fit(train_X)
            # train_X = sc.transform(train_X)
            test_X = all_X[index:index+4]
            # test_X = sc.transform(test_X)

            data.append(train_X)
            data.append(test_X)
            data.append(all_Y.drop([index, index+1, index+2, index+3]))
            data.append(all_Y[index:index+4])
            data_list.append(data)

        # 線形SVM
        # model = SVC(kernel='linear', random_state=None)
        # ロジスティック回帰
        model = LogisticRegression(random_state=None)

        for i, data in enumerate(data_list) :       
            model.fit(data[0], data[2])
            pred = model.predict(data[1])
            for j in range(len(pred)) :
                self.__results[pred[j]][data[3][i*4+j]] += 1

        # print(self.__accuracy())
        print(self.__results)
        # for i in show :
            # print(self.__precision(i))
            # print(self.__recall(i))
            # print(self.__f_value(i))

        return True

    def __accuracy(self) :
        t = self.__results[0][0] + self.__results[1][1] + self.__results[2][2]
        a = 0.0
        for results in self.__results :
            for result in results :
                a += result
        return t / a

    def __precision(self, i) :
        t = self.__results[i][i]
        a = self.__results[i][0] + self.__results[i][1] + self.__results[i][2]
        if a == 0 :
            return 0
        return t / a

    def __recall(self, i) :
        t = self.__results[i][i]
        a = self.__results[0][i] + self.__results[1][i] + self.__results[2][i]
        if a == 0 :
            return 0
        return t / a

    def __f_value(self, i) :
        p = self.__precision(i)
        r = self.__recall(i)
        if (r+p) == 0 :
            return 0

        return (2*r*p) / (r+p)

def main() :
    selects = ["ANSWER", "2", "11", "35", "83", "74"]
    svm = SVM('../feature_value/data/no_count/calm.csv', selects)
    svm.predict([0, 1, 2])

if __name__ == "__main__":
    main()