import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_curve, auc, accuracy_score)
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import linear_model
import gbdtree as gb
from sklearn.model_selection import GridSearchCV
import warnings
import pickle

class SplitMethod :

    @classmethod
    def cross_validation(data) :
        train_data = []
        test_data = []
        is_used = [False for i in range(len(data))]
        used_log = []

        for _ in range(len(data)) :
            test = []
            train = data
            for _ in range(4) :
                rand = int(random.random(0, len(data) - 1))
                while is_used[rand] :
                    rand = int(random.random(0, len(data) - 1))

                test.append(data[rand])
                train.remove(data[rand])
                used_log.append(rand)

            train_data.append(train)
            test_data.append(test)

        return train_data, test_data

    @classmethod
    def leave_human_out(data) :
        train_data = []
        test_data = []
        person_count = int(len(data) / 4 )
        for i in range(person_count) :
            train = data
            for j in range(4) :
                test_data.append(data[i * person_count + j])
                train.pop(i * person_count + j)
            train_data.append(train)
            test_data.append(test)

        return train_data, test_data

class RandomForest :
    def __init__(self, path, is_minimize=False, is_norm=False, file='', dir="") :
        self.__df = pd.read_csv(path)
        if is_minimize is True :
            self.__df['ANSWER'] = self.__df['ANSWER'].apply(lambda x: 0 if int(x) <= 2 else 1 )
            self.__size = 2
        else :
            self.__df['ANSWER'] = self.__df['ANSWER'].apply(lambda x: 0 if int(x) <= 2 else (2 if int(x) > 3 else 1))
            self.__size = 3

        # self.__df = self.__df.drop(['ID', 'SECTION'],axis=1)
        self.__results = [[0] * self.__size for i in range(self.__size)]

        self.__data_list = []
        self.__feature_list = []

        self.__params = []
        self.__is_norm = is_norm

        self.file = file

    # split_method is 'rand' or 'cross_validation' or 'leave_human_out'
    def predict(self, split_method='leave_human_out', seed=234, size=4) :
        warnings.filterwarnings('ignore')
        # [train_x, test_x, train_y, test_y]'s pattern list.
        data_list = []

        # clf = RandomForestClassifier(random_state=seed)
        clf = GridSearchCV(
            # LogisticRegression(random_state=seed),
            RandomForestClassifier(random_state=seed, class_weight='balanced'),
            self.__param(),
            cv=5,
            scoring='f1_macro'
        )

        # clf = linear_model.Lasso(alpha=14)
        # clf = linear_model.SGDRegressor(max_iter=1000)
        # clf = gb.GradientBoostedDT()

        all_X = self.__df.drop('ANSWER', axis=1)
        all_Y = self.__df.ANSWER

        # df_2 = pd.read_csv('../feature_value/data/kmean_2/swell.csv')
        # df_2 = df_2.drop(['ID', 'SECTION'], axis=1)
        # self.__df['ANSWER'] = self.__df['ANSWER'].apply(lambda x: 0 if int(x) <= 2 else (2 if int(x) > 3 else 1))
        # test_X, test_Y = self.__df.drop('ANSWER',axis=1), self.__df.ANSWER

        # clf = clf.fit(all_X, all_Y)
        # self.__feature_list.append(clf.feature_importances_)

        test_dataset = pd.read_csv('../feature_value/data/side_b_k2/' + self.file)
        test_dataset = test_dataset.drop(['ID', 'SECTION'], axis=1)

        if split_method == 'leave_human_out' :
            for i in range(int(len(self.__df) / 4)) :
                index = i * 4
                data = []
                if i+1 < len(self.__df) and self.__df.iloc[i]['ID'] is self.__df.iloc[i+1]['ID']:
                    data_list.append(self.__edit_dataset(index, all_X, all_Y, True, test_dataset))
                    i = i + 1
                else :
                    data_list.append(self.__edit_dataset(index, all_X, all_Y, False, test_dataset))
        else :
            (train_X, test_X, train_Y, test_Y) = train_test_split(all_X, all_Y, test_size=0.3, random_state=666)
            data_list.append([train_X, test_X, train_Y, test_Y])

        self.__results = [[0] * 3 for i in range(3)]

        total = 0
        for i, data in enumerate(data_list) :
            # print(data[0].isnull().sum())
            clf_fit = clf.fit(data[0].values, data[2].values)
            predictor = clf_fit.best_estimator_

            # TODO Save Models
            # filepath = "./result/model/" + str(self.__size) + "/" + dir + "_" + str(i) + ".sav"
            # pickle.dump(clf, open(filepath, 'wb'))

            self.__params.append(predictor.feature_importances_)
            pred = predictor.predict(data[1])
            for j in range(len(pred)) :
                self.__results[pred[j]][data[3][total]] += 1
                # self.__results[pred[j]][data[3][j]] += 1
                total = total + 1

        # f = open('./result/rf_feature.csv', 'w')
        # writer = csv.writer(f, lineterminator='\n')
        # writer.writerows(self.__feature_list)
        # f.close()

        return True

    def __edit_dataset(self, subject_index, baseX, baseY, is_pair, test_dataset) :
        result, X = [], baseX.drop(['ID', 'SECTION'],axis=1)
        test_X = test_dataset.drop('ANSWER', axis=1)
        test_Y = test_dataset.ANSWER.apply(lambda x: 0 if int(x) <= 2 else (2 if int(x) > 3 else 1))

        if self.__is_norm :
            sc = StandardScaler()
            sc.fit(X)
            X = pd.DataFrame(sc.transform(X), columns=X.columns)

        x = subject_index
        if is_pair :
            result.append(X.drop([x, x+1, x+2, x+3, x+4, x+5, x+6, x+7]))
            result.append(X.iloc[x:x+8])
            # result.append(test_X)
            result.append(baseY.drop([x, x+1, x+2, x+3, x+4, x+5, x+6, x+7]))
            result.append(baseY.iloc[x:x+8])
            # result.append(test_Y)
        else :
            result.append(X.drop([x, x+1, x+2, x+3]))
            result.append(X.iloc[x:x+4])
            # result.append(test_X)
            result.append(baseY.drop([x, x+1, x+2, x+3]))
            result.append(baseY.iloc[x:x+4])
            # result.append(test_Y)

        return result


    def get_accuracy(self) :
        x, y = 0, 0

        for i in range(len(self.__results)) :
            for j in range(len(self.__results[i])) :
                if i == j :
                    x += self.__results[i][j]
                y += self.__results[i][j]

        if y is 0 :
            return 0

        return (x / y)

    def get_precision(self, is_macro=False) :
        results, m = [], 0.0
        for i in range(len(self.__results)) :
            t = self.__results[i][i]
            a = 0.0
            for j in range(len(self.__results[i])) :
                a += self.__results[i][j]
            result = 0
            if a != 0 :
                result = t / a
            m += result
            results.append(result)

        if is_macro == True :
            m /= len(results)
            results.append(m)

        return results

    def get_recall(self, is_macro=False) :
        results, m = [], 0.0
        for i in range(len(self.__results)) :
            t = self.__results[i][i]
            a = 0.0
            for j in range(len(self.__results[i])) :
                a += self.__results[j][i]
            result = 0.0
            if a > 0.0 :
                result = t / a
            m += result
            results.append(result)

        if is_macro == True :
            m /= len(results)
            results.append(m)

        return results

    def get_f_measure(self, is_macro=False, is_others=False) :
        results = []
        r = self.get_recall(is_macro)
        p = self.get_precision(is_macro)
        for i in range(len(r)) :
            result = 0.0
            if (r[i] + p[i]) > 0.0 :
                result = 2 * r[i] * p[i] / (r[i] + p[i])

            results.append(result)

        if is_others == True :
            results.extend(r)
            results.extend(p)

        return results

    def get_all_results(self, is_macro=False) :
        results = []
        results.append(self.get_accuracy())
        results.extend(self.get_f_measure(is_macro, is_others=True))
        results.extend(self.__results)

        return self.__params, results

    def get_data_size(self) :
        return len(self.__df)

    def show_importance(self, data_index=0) :
        if self.__data_list == [] :
            print('Fail to show importance.')
            return False

        features = self.__data_list[data_index][0].columns
        importance = self.__data_list[data_index][4].feature_importance_
        indices = np.argsort(importance)

        # TODO たぶんこのままだと表示できない
        plt.figure(figsize=(6, 6))
        plt.barh(range(len(indices)), importance[indices], color='b', align='center')
        plt.yticks(range(len(indices)), features[indices])
        plt.show()

    def __param(self) :
        ret = {
            'n_estimators': np.arange(10, 50, 10),
            'criterion': ['gini', 'entropy'],
            # 'max_features': [None],
            'max_depth': np.arange(1, 10, 1),
            'min_samples_split': np.arange(4,5,1),
            # 'C': [10 ** i for i in range(-5, 6)],
        }

        return ret


def main() :
    label_file = '/bite.csv'
    dir = '/side_b_k1'
    forest = RandomForest('../feature_value/data/half/bite.csv')
    seeds = []
    with open('./result/seeds.csv', 'r') as f :
        reader = csv.reader(f)
        for row in reader :
            row = [int(x) for x in row]
            seeds.extend(row)

    result_path = './result/3_class' + dir + label_file
    a = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    # a = [800]
    # accs = []
    # f_values = []
    for i in a :
        forest.predict(seed=i)
    #     accs.append(acc)
    #     f_values.append(f_value)

    # t = 0.0
    # max_value = 0.0
    # print(accs)
    # for acc in accs :
    #     t += acc
    #     if max_value <= acc :
    #         max_value = acc
    # print('Accuracy')
    # print('Average:' + str(t / len(accs)))
    # print('Max:' + str(max_value))

    # t = 0.0
    # max_value = 0.0
    # print(f_values)
    # for f_value in f_values :
    #     t += f_value
    #     if max_value <= f_value :
    #         max_value = f_value
    # print('F_Measure')
    # print('Average:' + str(t / len(f_values)))
    # print('Max:' + str(max_value))
    return True

if __name__ == "__main__":
    main()