import umap, time, sys, os, glob
import pandas as pd
import warnings

warnings.simplefilter('ignore')

class OpensmileUmap:
    def __init__(self):
        self.__hoge = 0

    def opensmile_umap(self, n_components, is_update=False):
        print("OPENSMILE_UMAP")
        print(" Now Loading CSV...")
        df = pd.read_csv('../data/IS10_CD.csv')
        UMAP_CSVPATH = '../data/opensmile_umap/' + str(n_components) + ".csv"

        print(" Now Embedding...")
        if not os.path.exists(UMAP_CSVPATH) or is_update:
            # start_time = time.time()

            target_df = self.__df.drop('name', axis=1)
            df_concat = pd.concat([target_df, df])
            # self.__embedding = umap.UMAP(n_neighbors=int(len(df)/4),min_dist=0.1,n_components=n_components).fit(df.values)
            self.__embedding = umap.UMAP().fit(df_concat.values)
            # end_time = time.time()

            umap_df = pd.concat([self.__df['name'], pd.DataFrame(self.__embedding.embedding_[:len(target_df)])], axis=1)
            umap_df.to_csv(UMAP_CSVPATH)
        else:
            umap_df = pd.read_csv(UMAP_CSVPATH)

        print("     n_neighbors: " + str(int(len(df)/4)))
        print("     n_components: " + str(n_components))
            
        # total_values = [[0.0 for _ in range(n_components + 1)] for _ in range(4 * 48)]
        columns = ["ID", "SECTION", "COUNT"]
        features_df = pd.DataFrame(columns=columns)

        for i in range(n_components):
            columns.append("f" + str(i) + "_mean")
            columns.append("f" + str(i) + "_var")
            columns.append("f" + str(i) + "_max")
            columns.append("f" + str(i) + "_min")

        print(umap_df)

        for _, x in umap_df.iterrows():
            [subject, section, _] = [int(s) for s in x['name'].split("_")]
            # subject_num, section_num = int(subject) - 1, int(section)

            # while len(features_value) < subject_num + 1:
            #     features_value.append([[0.0 for _ in range(4 * n_components + 1)] for _ in range(4)])

            # features_value[subject_num][section_num][0] += 1
            # for i in range(n_components):
            #     value = x[str(i)]
            #     features_value[subject_num][section_num][4*i+1] += value

            #     if features_value[subject_num][section_num][4*i+3] < value:
            #         features_value[subject_num][section_num][4*i+3] = value

            #     if features_value[subject_num][section_num][4*i+4] > value:
            #         features_value[subject_num][section_num][4*i+4] = value

            if not (subject in features_df["ID"].values) or not(section in features_df["SECTION"].where(features_df["ID"] == subject).values):
                feature = pd.Series(index=columns, dtype='int64')

                feature["ID"], feature["SECTION"] = subject, section
                for i in range(n_components):
                    feature["f" + str(i) + "_min"] = 9999.0
                    feature["f" + str(i) + "_max"] = -9999.0

                features_df = features_df.append(feature, ignore_index=True)

            target = (features_df["ID"] == subject) & (features_df["SECTION"] == section)
            features_df.loc[target, "COUNT"] += 1

            for i in range(n_components):
                value = x[i]

                features_df.loc[target, "f" + str(i) + "_mean"] += value
                if features_df.loc[target, "f" + str(i) + "_min"].values[0] > value:
                    features_df.loc[target, "f" + str(i) + "_min"] = value
                if features_df.loc[target, "f" + str(i) + "_max"].values[0] < value:
                    features_df.loc[target, "f" + str(i) + "_max"] = value
                    
        for i, _ in features_df.iterrows():
            for j in range(n_components):
                if features_df["f" + str(j) + "_mean"][i] != 0.0:
                    features_df["f" + str(j) + "_mean"][i] /= features_df["COUNT"][i]

        for _, x in umap_df.iterrows():
            [subject, section, _] = [int(s) for s in x['name'].split("_")]
            target = (features_df["ID"] == subject) & (features_df["SECTION"] == section)

            for i in range(n_components):
                value = x[i] - features_df.loc[target, "f" + str(i) + "_mean"].values[0]

                features_df.loc[target, "f" + str(i) + "_var"] += value * value

        for i, _ in features_df.iterrows():
            for j in range(n_components):
                if features_df["f" + str(j) + "_var"][i] != 0.0:
                    features_df["f" + str(j) + "_var"][i] /= features_df["COUNT"][i] 
        
        return features_df.drop("COUNT", axis=1)

        # print(pd.DataFrame(features_value, columns=columns[2:]))

        # features_df = pd.concat([features_df, pd.DataFrame(features_value, columns=columns[2:])], axis=1)

        # print(features_df)


        # for _, x in umap_df.iterrows():
        #     [subject, section, _] = x['name'].split("_")

        #     if not (int(subject) in features["ID"].values) or not(int(section) in features["SECTION"].where(features["ID"] == int(subject)).values):
        #         feature = pd.Series(index=columns, dtype='int64')

        #         feature["ID"], feature["SECTION"] = int(subject), int(section)
        #         features = features.append(feature, ignore_index=True)

        # for i in range(n_components):
        #     features["f" + str(i) + "_mean"] = 0.0
        #     features["f" + str(i) + "_var"] = 0.0
        #     features["f" + str(i) + "_max"] = 0.0
        #     features["f" + str(i) + "_min"] = 0.0
        #     # columns.append("f" + str(i) + "_mean")
        #     # columns.append("f" + str(i) + "_var")
        #     # columns.append("f" + str(i) + "_max")
        #     # columns.append("f" + str(i) + "_min")

        # features["COUNT"] = 0

        # for _, x in umap_df.iterrows():
        #     [subject, section, _] = x['name'].split("_")
        #     target_df = features[(features["ID"] == int(subject)) & (features["SECTION"] == int(section))]

        #     for i in range(n_components):
        #         target_df["f" + str(i) + "_mean"] += x[str(i)]

        # print(features)
            
            # for i in range(n_components):
            #     # total_values[feature_id][i] += x[str(i)]
            
            # total_values[feature_id][n_components] += 1

        # print(total_values)

        # for section in total_values:
        #     for feature in range(n_components):
        #         if section[feature] != 0.0:
        #             section[feature] /= section[n_components]

        # print()
        # print(total_values)

    def __read_all_subject(self):
        self.__df = pd.read_csv("../data/solo_opensmile/csv/part.csv")
        for i, name in enumerate(self.__df['name']):
            name = name.replace("../../processing/data/state_sound/default/1/", "")
            group_id = name[:name.find("/")]
            name = name.replace(group_id, "").replace(".wav", "")
            subject_id = group_id.split("_")

            subject_index = subject_id[int(name[1])] if len(subject_id) == 2 else subject_id[0]
            section_index = name[3]
            active_index = name[5:]

            self.__df['name'][i] = subject_index + "_" + section_index + "_" + active_index

    def get_df(self, n_components=2, is_update=False):
        self.__read_all_subject()
        return self.opensmile_umap(n_components=n_components, is_update=is_update)

if __name__ == "__main__":
    OU = OpensmileUmap()
    # OU.opensmile_umap()

    # TODO(Bag Fix): csvの読み込みと書き込みの時とで型が違う？
    umap_df = OU.get_df(is_update=True)
    vf_pathes = glob.glob("../data/JSKE_2020_base/*.csv")

    if not os.path.exists("../data/OpenSmile_umap_2"):
        os.mkdir("../data/OpenSmile_umap_2")

    for path in vf_pathes:
        vf_df = pd.read_csv(path)
        print(vf_df)
        features_df = pd.concat([vf_df, umap_df], axis=1)
        print(features_df)
        features_df.to_csv("../data/OpenSmile_umap_2/bite.csv")
        umap_df.to_csv("../data/OpenSmile_umap_2/test.csv")
        sys.exit()
