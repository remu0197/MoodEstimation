## 雰囲気評価と相手の評価，相手の評価と性格評価というようにそれぞれ相関分析を行う！
## スライドをまとめる

import pandas as pd 
import seaborn
import matplotlib.pyplot as plt

class CorrelationAnalizer :
    def __init__(self, path='') :
        self.DATAPATH = '../feature_value/questionaire/all_.csv'
        if path is not '' :
            self.DATAPATH = path
        self.__corr_mat = []
        self.__LABEL_COLUMN = [
            'Swell',
            'Serious',
            'Bite',
            'Cheerful',
            'Calm',
            'Friendly',
            'Eminent'
        ]
        self.__IMPRESSION_COLUMN = [
            'Syako',
            'Talkable',
            'Yasashi',
            'Kigaau'
        ]
        

    def get_whole_corr(self, method='spearman', is_show=True) :
        df = pd.read_csv(self.DATAPATH).iloc[0:45]
        self.__corr_mat = df.corr(method)

        show_df = pd.DataFrame(columns=self.__IMPRESSION_COLUMN)

        for label in self.__LABEL_COLUMN :
            for i in range(3,4) :
                temp = self.__corr_mat.loc[label + '_' + str(i+1), 'Syako':'Kigaau']
                show_df = show_df.append(temp)

        if is_show is True :
            seaborn.heatmap(
                show_df,
                vmin=-1.0,
                vmax=1.0,
                center=0,
                annot=True,
                fmt='.1f',
            )

            plt.show()

        return self.__corr_mat

    def get_summary_corr(self, method='pearson', is_show=True) :
        df = pd.read_csv(self.DATAPATH).iloc[35:45]
        summary_df = pd.DataFrame(columns=self.__LABEL_COLUMN+self.__IMPRESSION_COLUMN)

        for i in range(1, 5) :
            for _, row in df.iterrows() :
                x = row['Syako':'Kigaau']
                for label in self.__LABEL_COLUMN:
                    value = row[label + '_' + str(i)]
                    temp = pd.Series({label: value})
                    x = x.append(temp)

                summary_df = summary_df.append(x, ignore_index=True)

        self.__corr_mat = summary_df.corr(method)
        show_df = self.__corr_mat.loc['Swell':'Eminent', 'Swell':'Eminent']

        if is_show is True :
            seaborn.heatmap(
                show_df,
                vmin=-1.0,
                vmax=1.0,
                center=0,
                annot=True,
                fmt='.1f',
            )

            plt.show()

        return self.__corr_mat

if __name__ == '__main__' :
    CA = CorrelationAnalizer() 
    CA.get_whole_corr()