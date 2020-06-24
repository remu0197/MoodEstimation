# import pandas as pd 
# import matplotlib.pyplot as plt 
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import (roc_curve, auc, accuracy_score)
# import pydotplus
# from IPython.display import Image
# from graphviz import Digraph
# from sklearn.externals.six import StringIO
# import sklearn.tree
# from sklearn.ensemble import RandomForestClassifier
# import numpy as np 

# col_names = ['PassengerId','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']
# df = pd.read_csv('./train.csv')
# # sns.countplot('Sex',hue='Survived',data=df)

# # 欠損値処理 
# #   欠損値：空白になってる値
# #       ⇒　空白を埋める処理
# df['Fare'] = df['Fare'].fillna(df['Fare'].median())
# df['Age'] = df['Age'].fillna(df['Age'].median())
# df['Embarked'] = df['Embarked'].fillna('S')

# # カテゴリ変数の変換
# df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)
# df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

# # 前処理
# df = df.drop('PassengerId', axis=1)
# df = df.drop('Cabin', axis=1)
# train_x = df.drop('Survived', axis=1)
# train_y = df.Survived
# (train_x, test_x, train_y, test_y) = train_test_split(train_x, train_y, test_size=0.3, random_state=666)

# print(df)

# # 決定木
# #   DecisionTreeClassifier:
# #       criterion           - 分割基準．gini or entropy
# #       max_depth           - 木の深さ．
# #       max_features        - 最適な分割をする際の特徴量の数
# #       min_samples_split   - 分割する際のサンプル数
# #       random_state        - seed値

# # clf = DecisionTreeClassifier(random_state=0)
# # clf = clf.fit(train_x, train_y)

# # # 学習
# # pred = clf.predict(test_x)
# # fpr, tpr, thresholds = roc_curve(test_y, pred, pos_label=1)
# # auc(fpr, tpr)
# # accuracy_score(pred, test_y)

# # dot_data = StringIO()
# # sklearn.tree.export_graphviz(clf, out_file=dot_data, feature_names=train_x.columns, max_depth=3)
# # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# # graph.write_pdf("graph.pdf")
# # Image(graph.create_png())

# # ランダムフォレスト
# #   RandomForestClassifier
# #       n_estimators        - 木をいくつ生成するか．デフォルトは10
# #       max_depth           - 同上
# #       max_features        - 同上
# #       min_sample_split    - 同上
# #       random_state_seed   - 同上
# clf = RandomForestClassifier(random_state=0)
# clf = clf.fit(train_x, train_y)
# pred = clf.predict(test_x)
# fpr, tpr, thresholds = roc_curve(test_y, pred, pos_label=1)
# print(auc(fpr, tpr))
# print(accuracy_score(pred, test_y))

# features = train_x.columns
# importance = clf.feature_importances_
# indices = np.argsort(importance)

# plt.figure(figsize=(6, 6))
# plt.barh(range(len(indices)), importance[indices], color='b', align='center')
# plt.yticks(range(len(indices)), features[indices])
# plt.show()

import pyper
import pandas as pd
import numpy as np 

# df = pd.DataFrame({'seg':['segA','segB','segC','segA','segB','segC'],
#                   'y':[0.805389,0.912482,-1.081869,-0.245571,-0.851627,0.090373],
#                   'x1':[-1.226449,1.139922,-1.138730,-0.135973,-2.145796,-0.529889],
#                   'x2':[-0.310213,-1.138528,-0.265708,0.182349,0.382971,-0.234263],
#                   'x3':[0.712231,0.285189,0.947262,0.346676,-0.386438,-0.288878],
#                   'x4':[0.131545,2.123124,0.939928,2.254102,0.143505,1.039283]
#                   })

df = pd.read_csv('../feature_value/data/all_x/calm.csv')
# df['ANSWER'] = df['ANSWER'].apply(lambda x: 0 if int(x) < 3 else 1)
df['ANSWER'] = df['ANSWER'].apply(lambda x: 0 if int(x) <= 2 else(2 if int(x) > 3 else 1))
df = df.drop(['ID', 'SECTION'],axis=1)

labels = "ANSWER~x1"
for i in range(2, 115) :
    labels += "+x" + str(i)

# R のインスタンスを作る
r = pyper.R(use_pandas='True')
r.assign("data", df)

s = "result <- lm(" + labels + ", data=data)"

#print(r('summary(result)'))

r("nullModel <- lm(ANSWER~1, data=data)")
r("fullModel <- lm(" + labels + ", data=data)")

r("stepModel <- step(nullModel,scope=list(lower=nullModel,upper=fullModel),direction='both')")
print(r("summary(stepModel)"))

print("回帰係数")
print(r.get('stepModel$coefficients'))

print("P-values")
print(r.get('summary(stepModel)$coefficients[,"Pr(>|t|)"]'))

print("決定係数")
print(r.get('summary(stepModel)$r.squared'))