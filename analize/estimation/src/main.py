import sys
import glob
import random
import csv
import os
from random_forest import RandomForest
import numpy as np
import warnings
import datetime
import concurrent.futures
import os
from mood_estimater import MoodEstimator

def main(dir, method, trials, is_minimize, is_norm) :
    seeds = []
    used_slead_count, MAX_SLEAD_COUNT = 0, 24

    with open('../result/seeds.csv', 'r') as f :
        reader = csv.reader(f)
        for row in reader :
            row = [int(x) for x in row]
            seeds.extend(row)

    TRIALS_COUNT = int(trials)
    if len(seeds) < TRIALS_COUNT :
        for _ in range(TRIALS_COUNT - len(seeds)) :
            x = random.randint(0, TRIALS_COUNT * 10)
            while x in seeds :
                x = random.randint(0, TRIALS_COUNT * 10)
            seeds.append(x)
    elif len(seeds) > TRIALS_COUNT :
        seeds = seeds[:-(len(seeds) - TRIALS_COUNT)]

    with open('../result/seeds.csv', 'w') as f :
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(seeds)

    taxonomy = '2_class/' if bool(is_minimize) is True else '3_class/'
    outdir = dir + ('_norm' if bool(is_norm) is True else '')

    result_path = '../result/' + taxonomy + outdir + '/'
    print('result path: ' + result_path)
    if not os.path.isdir(result_path) :
        os.mkdir(result_path)
    result_path = result_path + '{0:%y%m%d%H%M}'.format(datetime.datetime.now()) + '/'
    if not os.path.isdir(result_path) :
        os.mkdir(result_path)

    executer = concurrent.futures.ProcessPoolExecutor(max_workers=MAX_SLEAD_COUNT)

    if method == 'random_forest' :
        target_path = '../../feature_value/data/' + dir
        pathes = glob.glob(target_path + '/*.csv')
        files = os.listdir(target_path)
        result_all = [[] for i in range(len(pathes))]
        
        for i, path in enumerate(pathes) :
            print('Current Target: ' + path)
            forest = RandomForest(path, bool(is_minimize), bool(is_norm), files[i], dir)
            select_futures = [[] for _ in range(7)]
            features = []
            result_all[i].append([])
            result_all[i].append([['datasize', str(forest.get_data_size())], [], ['ID', 'Accuracy', 'f_0', 'f_1', 'f_2', 'f_m', 'r_0', 'r_1', 'r_2', 'r_m', 'p_0', 'p_1', 'p_2', 'p_m', '+1', '0', '-1']])

            j = 0
            while j < len(seeds) or len(features) > 0:
                if used_slead_count < MAX_SLEAD_COUNT and j < len(seeds):
                    features.append(executer.submit(
                        update_result_list, 
                        i, 
                        j, 
                        forest, 
                        path, 
                        result_all,
                        result_path + files[i],
                        select_futures,
                        seeds[j],
                    ))
                        
                    used_slead_count += 1
                    j += 1
                
                if used_slead_count is MAX_SLEAD_COUNT: 
                    print('     Waited')
                    for f in concurrent.futures.as_completed(features) :
                        # p, r = f.result()
                        # result_all[i][1].append(r)
                        # result_all[i][0].append(p)
                        f.result()
                        used_slead_count -= 1
                        features.remove(f)
                
        for i in range(len(pathes)):
            with open(result_path + files[i], "w") as f :
                writer = csv.writer(f, lineterminator='\n')
                writer.writerows(result_all[i][0])
                writer.writerows(result_all[i][1])
                print("Complete: " + result_path + files[i])

        features_path = result_path + "/features/"
        if not os.path.exists(features_path):
            os.mkdir(features_path)

        for i, name in enumerate(files):
            with open(features_path + name, "w") as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerows(select_futures[i])

    else :
        print('Error: can\'t estimate with \'' + method + '\'')
        return False

    return True

def update_result_list(path_index, index, estimator, path, result_all, result_path, select_futures, seed=0) :
    print(' seed ID: ' + str(index) + ' (' + path + ')')
    result, param = [index], [index]
    estimator.predict(seed=seed)
    p, r = estimator.get_all_results(is_macro=True)
    result.extend(r)
    param.extend(p)
    result = [str(x) for x in result]

    result_all[path_index][0].append(param)
    result_all[path_index][1].append(result)

    select_futures[path_index].append(estimator.get_select_features())

    print('     Complete: result_path ' + str(len(result_all[path_index][0])))
        
    # return result, param

def change_no(x) :
    for i in x :
        print(func(i))

def func(x) :
    if x <= 20 :
        if x <= 16 :
            return x
        else :  
            return x + 4
    else :
        a = x - 21
        b = int(a / 5)
        c = a % 5
        if c == 4 :
            return 24 + (b * 6) + 6
        else :
            return 24 + (b * 6) + c + 1

def argument_error(error_string) :
    print(error_string)
    print('     0: learning datasets\' directory name')
    # print('     1: cluster data\'s directory name')
    print('     2: estimation method name')
    print('     3: trials count')
    print('     4: Flag of is_minimize(0 or 1)')
    print('     5: Flag of is_norm(0 or 1)')
    sys.exit()

if __name__ == "__main__":
    # args = sys.argv
    # warnings.filterwarnings('always')
    # ARG_LENGTH = 6

    # if len(args) < ARG_LENGTH :
    #     argument_error('Error: few argments')
    # elif len(args) > ARG_LENGTH :
    #     argument_error('Error: too many argments')

    # main(args[1], args[2], args[3], int(args[4]), int(args[5]))  

    me = MoodEstimator(
        trials_count=10,
        dirname="JSKE_2020_base",
        is_minimize=True,
        is_norm=True,
    )

    me.predict(method="random_forest")