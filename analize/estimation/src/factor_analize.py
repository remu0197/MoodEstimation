import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis

class FactorAnalize :
    def __init__(self, path) :
        self.__df = pd.read_csv(path)

        self.__df = self.__df.drop(['ID', 'SECTION'],axis=1)
        # change answer from 5 points to 3 points.
        self.__df['ANSWER'] = self.__df['ANSWER'].apply(lambda x: 0 if int(x) <= 2 else (2 if int(x) > 3 else 1))
        # del self.__df['ANSWER']

        # Standardization of Input Data
        sc = StandardScaler()
        self.__input = sc.fit_transform(self.__df.values)
        # self.__df['ANSWER'] = answer

    def predict(self, n_components=5, random_state=0) :
        factor = FactorAnalysis(n_components=n_components, random_state=random_state).fit(self.__input)
        print(factor.components_)




if __name__ == "__main__":
    factor = FactorAnalize('../feature_value/data/no_count/eminent.csv')
    factor.predict()