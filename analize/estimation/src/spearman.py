import pandas as pd 

def test() :
    x = [1, 2, 3, 4, 5, 6, 7]
    y = [3, 6, 2, 4, 0, 1, 2]
    z = [3, 2, 1, 6, 7, 8, 6]

    df = pd.DataFrame({'x': x, 'y': y, 'z': z})
    print(df.corr(method="spearman")['x'])

def mood_and_personality() :
    filepath = "../feature_value/questionaire/all.csv"
    targetpath = "./result/spearman_m+p.csv"
    targetpath2 = "./result/spearman_m.csv"
    df = pd.read_csv(filepath)
    result = pd.DataFrame()
    
    for j in range(1, 8) :
        for i in range(1, 5) :
            len_name = str(i) + '_' + str(j)
            target = df.loc[:,[len_name, '5', '6', '7', '8']]
            result[len_name] = target.corr(method="spearman")[len_name]

    result.to_csv(targetpath)

    ##TODO mood spearman

def main() :
    mood_and_personality()

if __name__ == "__main__":
    mood_and_personality()