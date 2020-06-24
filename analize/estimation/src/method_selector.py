import csv, random, os, glob
import concurrent.futures
import pandas as pd 
from sklearn.model_selection import train_test_split

class MethodSelector :
    def __init__(self) :
        self.__seeds = []
        self.__dataset_info = pd.DataFrame({
            'is_clusterized': [],
            'data_sets': [],
            'result_dirs': []})
        self.__used_slead_count = 0
        self.__MAX_SLEAD_COUNT = os.cpu_count() - 1
        self.__datasets = []

        with open('./result/seeds.csv', 'r') as f :
            reader = csv.reader(f)
            for row in reader :
                row = [int(x) for x in row]
                self.__seeds.extend(row)

        self.__datasets = []

    def __read_csv(self, path) :
        result = []
        with open(path, 'r') as f :
            reader = csv.reader(f)
            for row in reader :
                result.append(row)
        
        return result

    def __clusterized_dataset(self, cluster_path, df_all) :
        cluster_info = self.__read_csv(cluster_path)[0]
        cluster_info = [int(x) for x in cluster_info]
        dataset_list = [pd.DataFrame(columns=df_all.columns) for _ in range(max(cluster_info)+1)]

        for participant_ID, cluster_ID in enumerate(cluster_info) :
            bi = participant_ID * 4
            if bi > len(df_all) :
                break
            dataset_list[cluster_ID] = dataset_list[cluster_ID].append(df_all.iloc[bi:bi+4])

        return dataset_list

    def __split_dataset(self, split_method, df_all) :
        dataset_list = []

        if split_method is 'leave_one_person_out' :
            current_ID = int(df_all.iloc[0]['ID'])
            test_df, df_temp = pd.DataFrame(columns=df_all.columns), df_all
            for index, row in df_all.iterrows():
                if current_ID != int(row['ID']) :
                    current_ID = int(row['ID'])
                    train_x, train_y = df_temp.drop('ANSWER', axis=1), df_temp.ANSWER
                    test_x, test_y = test_df.drop('ANSWER', axis=1), test_df.ANSWER
                    dataset_list.append([train_x, test_x, train_y, test_y])

                    df_temp = df_all
                    test_df = pd.DataFrame(columns=df_all.columns)

                test_df = test_df.append(row)
                df_temp = df_temp.drop(index)

        elif split_method is 'leave_one_group_out' :
            pair_list = self.__read_csv('../feature_value/data/pair.csv')
            current_pair = pair_list[0]

            current_ID = int(df_all.iloc[0]['ID'])
            test_df, df_temp = pd.DataFrame(columns=df_all.columns), df_all
            for index, row in df_all.iterrows():
                if current_ID != int(row['ID']) :
                    current_ID = int(row['ID'])

                if not str(current_ID) in current_pair :
                    train_x, train_y = df_temp.drop('ANSWER', axis=1), df_temp.ANSWER
                    test_x, test_y = test_df.drop('ANSWER', axis=1), test_df.ANSWER
                    dataset_list.append([train_x, test_x, train_y, test_y])

                    df_temp = df_all
                    test_df = pd.DataFrame(columns=df_all.columns)

                    for pair in pair_list :
                        if str(current_ID) in pair :
                            current_pair = pair
                            break

                test_df = test_df.append(row)
                df_temp = df_temp.drop(index)
            
        else :
            base_x, base_y = df_all.drop('ANSWER', axis=1), df_all.ANSWER
            (train_X, test_X, train_Y, test_Y) = train_test_split(base_x, base_y, test_size=0.2, random_state=666)
            dataset_list.append([train_X, test_X, train_Y, test_Y])
            
        return dataset_list

    def set_dataset(self, dataset_dir, cluster_dir, split_method, is_minimize=True, is_norm=True) :
        target_dir = '../feature_value/data/' + dataset_dir
        target_pathes = glob.glob(target_dir + '/*.csv')
        cluster_pathes = glob.glob('./clustering/' + cluster_dir + '/*.csv')
        datasets = pd.DataFrame(columns=['Outpath', 'Datasets'])

        taxonomy = '2_class/' if bool(is_minimize) is True else '3_class/'
        outdir = dataset_dir + ('_norm' if bool(is_norm) is True else '')
        outpath = './result/' + taxonomy + outdir + '/'

        for path in target_pathes :
            print(path)
            df = pd.read_csv(path)
            target_datasets = pd.DataFrame(columns=['ID', 'Dataset'])
            if is_minimize is True :
                df['ANSWER'] = df['ANSWER'].apply(lambda x: 0 if int(x) <= 2 else 1 )
            else :
                df['ANSWER'] = df['ANSWER'].apply(lambda x: 0 if int(x) <= 2 else (2 if int(x) > 3 else 1))

            for cluster_path in cluster_pathes :
                clusters = self.__clusterized_dataset(cluster_path, df)
                for i, cluster in enumerate(clusters) :
                    dataset_info = pd.Series(
                        [os.path.basename(cluster_path) + '_' + str(i), self.__split_dataset(split_method, cluster)],
                        index=target_datasets.columns
                    )
                    target_datasets = target_datasets.append(dataset_info, ignore_index=True)

            dataset = pd.Series(
                [outpath + os.path.basename(path), target_datasets],
                index=datasets.columns
            )
            datasets = datasets.append(dataset, ignore_index=True)

        self.__datasets.append(datasets, ignore_index=True)

    def predict(self, method) :
        # executer = concurrent.futures.ThreadPoolExecutor(max_workers=self.__MAX_SLEAD_COUNT)
        for datasets in self.__datasets :
            for dataset in datasets :
                print(dataset)

def main() :
    MS = MethodSelector()
    MS.set_dataset('all','20191205','leave_one_group_out')

    MS.predict('random_forest')

if __name__ == "__main__" :
    main()  