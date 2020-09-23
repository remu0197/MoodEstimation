# from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import pandas as pd
import glob, os, csv, random, datetime, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from random_forest import RandomForest

class MoodEstimator:
    def __init__(self, trials_count, dirname, is_minimize=False, is_norm=False):
        self.__used_process_count = 0
        self.__MAX_PROCESS_COUNT = 20
        self.__IS_MINIMIZE = is_minimize
        self.__IS_NORM = is_norm
        self.__DIR_NAME = dirname
        self.__file_names = []

        self.__y_preds = [[] for _ in range(7)]
        self.__y_trues = [[] for _ in range(7)]

        self.__seeds = []
        with open("../result/seeds.csv", "r") as f:
            reader = csv.reader(f)
            for row in reader:
                row = [int(x) for x in row]
                self.__seeds.extend(row)

        self.__TRIALS_COUNT = int(trials_count)
        if len(self.__seeds) < self.__TRIALS_COUNT :
            for _ in range(self.__TRIALS_COUNT - len(self.__seeds)) :
                x = random.randint(0, self.__TRIALS_COUNT * 10)
                while x in self.__seeds :
                    x = random.randint(0, self.__TRIALS_COUNT * 10)
                self.__seeds.append(x)

        elif len(self.__seeds) > self.__TRIALS_COUNT :
            self.__seeds = self.__seeds[:-(len(self.__seeds) - self.__TRIALS_COUNT)]

        with open('../result/seeds.csv', 'w') as f :
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(self.__seeds)

    def __edit_dataset(self, path, method="leave_human_out"):
        df = pd.read_csv(path)
        all_X, all_Y = df.drop(['ANSWER', 'ID', 'SECTION'], axis=1), df.ANSWER
        datasets = [path]

        if self.__IS_MINIMIZE:
            all_Y = all_Y.apply(lambda x: 0 if int(x) <= 2 else 1)
        else:
            all_Y = all_Y.apply(lambda x: 0 if int(x) <= 2 else (2 if int(x) > 3 else 1))

        if self.__IS_NORM:
            sc = StandardScaler()
            sc.fit(all_X)
            all_X = pd.DataFrame(sc.transform(all_X), columns=all_X.columns)

        if method == "leave_human_out":
            for i in range(int(len(df)/ 4)):
                x = i * 4
                dataset = []

                dataset.append(all_X.drop([x, x+1, x+2, x+3]))
                dataset.append(all_X.iloc[x:x+4])
                dataset.append(all_Y.drop([x, x+1, x+2, x+3]))
                dataset.append(all_Y.iloc[x:x+4])

                datasets.append(dataset)

        return datasets

    def __predict(self, method, dataset, target_index, seed):
        if method == "random_forest":
            rf = RandomForest()
            y_pred, y_true = rf.predict(dataset, seed)
            self.__y_preds[target_index].extend(y_pred)
            self.__y_trues[target_index].extend(y_true)
                
    def predict(self, method):
        warnings.filterwarnings('ignore')
        pathes = glob.glob("../../feature_value/data/" + self.__DIR_NAME + "/*.csv")
        target_datasets = []
        for path in pathes:
            filename = os.path.basename(path)
            self.__file_names.append(filename)
            target_datasets.append(self.__edit_dataset(path=path))
        
        executer = ProcessPoolExecutor(max_workers=self.__MAX_PROCESS_COUNT)
        futures = []

        for i, dataset in enumerate(target_datasets):
            seed_index = 0

            while seed_index < len(self.__seeds) or len(futures) > 0:
                if self.__used_process_count < self.__MAX_PROCESS_COUNT:        
                    print("Set Target: " + dataset[0] + " - " + str(seed_index))
                    futures.append(executer.submit(
                        self.__predict,
                        method,
                        dataset[1:],
                        i,
                        self.__seeds[seed_index]
                    ))
                    self.__used_process_count += 1
                    seed_index += 1
                else:
                    for f in as_completed(futures):
                        f.result()
                        self.__used_process_count -= 1
                        futures.remove(f)

    def output_result(self):
        classdir = '2_class/' if bool(self.__IS_MINIMIZE) else '3_class/'
        outdir = self.__DIR_NAME + ('_norm' if bool(self.__IS_NORM) is True else '')
        result_path = "../result/" + classdir + outdir + "/"
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        
        result_path = result_path + '{0:%y%m%d%H%M}'.format(datetime.datetime.now()) + '/'
        if not os.path.isdir(result_path) :
            os.mkdir(result_path)

        for i, filename in enumerate(self.__file_names):
            d = classification_report(self.__y_trues[i], self.__y_preds[i], output_dict=True)
            df = pd.DataFrame(d)
            df.to_csv(result_path + filename)