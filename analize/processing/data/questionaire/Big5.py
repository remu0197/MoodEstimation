import csv
import glob

class big5 :
    def __init__(self) :
        self.__words = []
        self.__f = []

    def load_factors(self, filepath) :
        csv_file = open(filepath, "r")
        data = csv.reader(csv_file)

        for row in data :
            self.__words.append(row[0])
            self.__f.append(row[1:6])

        return

    def apply_qustionnaire(self, q_filepath, r_filepath, p_filepath = '../data/big5.csv') :
        result = []

        questionnaire_file = open(q_filepath, "r")
        questionnaire_data = csv.reader(questionnaire_file)
        parameters_file = open(p_filepath, "r")
        parameters = csv.reader(parameters_file) 

        for data in questionnaire_data :
            x = []
            x.extend(data[0:3])
            x.extend([0, 0, 0, 0, 0])
            
            for answer in data[3:] :
                for row in parameters :
                    for i in range(1, 5) :
                        value = row[i]
                        if answer == '全く当てはまらない' :
                            x[i + 2] += 1 * value
                        elif answer == '当てはまらない' :
                            x[i + 2] += 2 * value
                        elif answer == 'やや当てはまらない' :
                            x[i + 2] += 3 * value
                        elif answer == 'どちらでもない' :
                            x[i + 2] += 4 * value
                        elif answer == 'やや当てはまる' :
                            x[i + 2] += 5 * value
                        elif answer == '当てはまる' :
                            x[i + 2] += 6 * value 
                        elif answer == '非常に当てはまる' :
                            x[i + 2] += 7 + value

            result.append(x)

        result_file = open(r_filepath, "w")
        writer = csv.writer(result_file, lineterminator='\n')
        writer.writerows(result)
        return 

if __name__ == '__main__': 
    # example of execute VAD
    data = big5()
    data.load_factors('../data/big5.csv')
    data.apply_qustionnaire('../data/personality.csv', '../data/big5_results.csv')