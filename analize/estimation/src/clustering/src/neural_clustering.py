import pandas as pd
from sklearn.neural_network import MLPClassifier

class NeuralClustering :
    def __init__(self, path, is_mizimize=True) :
        self.__df = pd.read_csv(path)
        self.__df['ANSWER'] = self.__df['ANSWER'].apply(lambda x: 0 if int(x) <= 2 else (2 if int(x) > 3 else 1))
        self.__df = self.__df.drop(['ID', 'SECTION'],axis=1)
        
    def test(self) :
        clf = MLPClassifier(hidden_layer_sizes=(5,),random_state=0)
        test_x = self.__df.drop('ANSWER', axis=1)
        test_y = self.__df.ANSWER
        clf.fit(test_x,test_y)

        print('fin:')
        print(clf.get_params())