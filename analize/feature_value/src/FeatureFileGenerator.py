import csv
import os
import pandas as pd 

class FeatureFileGenerator :
    def __init__(self, dir, label_filename='') :
        self.__file_pathes = []
        # self.__IS_LABELED = (label_filename != '')
        self.__col_name = ['ID', 'SECTION']
        self.__TARGET_DIR = '../data/' + str(dir) + '/'
        self.__generater_logs = []
        self.__datasets = [pd.DataFrame(columns=['ID','SECTION'])] * 7
        self.__IS_LABELED = label_filename != ''
    
        s = pd.Series([1, 1], index=self.__dataset.columns, name=0)
        # print(self.__dataset.append(s))

        if self.__IS_LABELED:
            # read label_info and generate each files of labels
            with open('../resource/label_info.csv', 'r') as f:
                reader = csv.reader(f)
                for row in reader :
                    path = self.__TARGET_DIR + str(row[0]) + '.csv'
                    self.__file_pathes.append(path)

            # add label_info's colmun
            self.__col_name.append('ANSWER')
            
            with open('../questionaire/all.csv', 'r') as f:
                reader = csv.reader(f)
                for row in reader :
                    

        else :
            # generate file without label_info
            path = self.__TARGET_DIR + 'no_labeled.csv'
            self.__file_pathes.append(path)

        

    def set_UIF(self) :
        a = 0

    def set_HP(self) :
        b = 0

    def generate(self) :
        if not os.path.exists(self.__TARGET_DIR) :
            os.mkdir(self.__TARGET_DIR)

        # for path in self.__file_pathes :



if __name__ == "__main__":
    generator = FeatureFileGenerator('ajfoep', True)
    # generator.generate()