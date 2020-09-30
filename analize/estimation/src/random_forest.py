from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os, sys

class RandomForest:
    def __param(self) :
        ret = {
            'n_estimators': np.arange(100,101,1),
            'criterion': ['gini', 'entropy'],
            'max_depth': np.arange(6, 60, 6),
            'min_samples_split': np.arange(4,5,1),
            'min_samples_leaf': np.arange(4, 40, 4),
            
            # For SVM: 
            # 'max_features': [None],
            # 'C': [10 ** i for i in range(-5, 6)],
        }

        return ret

    def predict(self, dataset, seed=123):
        # executer = ThreadPoolExecutor(max_workers=self.__MAX_THREAD_COUNT)
        # futures = []
        # y_pred, y_true = [], []

        # dataset_index = 0
        # while dataset_index < len(dataset) or len(futures) > 0:
            # if self.__used_thread_count < self.__MAX_THREAD_COUNT:
            #     clf = GridSearchCV(
            #         RandomForestClassifier(
            #             random_state=seed, 
            #             class_weight='balanced'
            #         ),
            #         self.__param(),
            #         cv=34,
            #         scoring='f1_macro'
            #     )

        # print(seed)

        clf = GridSearchCV(
            RandomForestClassifier(random_state=seed, class_weight='balanced'),
            self.__param(),
            cv=34,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=1,
        )

        clf_fit = clf.fit(dataset[0].values, dataset[2].values)
        predictor = clf_fit.best_estimator_
        pred = predictor.predict(dataset[1])

        return pred, dataset[3]
