from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os

class RandomForest:
    def __init__(self):
        self.__used_thread_count = 0
        self.__MAX_THREAD_COUNT = min(32, os.cpu_count() + 4)

    def __param(self) :
        ret = {
            'n_estimators': np.arange(20, 120, 20),
            'criterion': ['gini', 'entropy'],
            # 'max_features': [None],
            'max_depth': np.arange(6, 60, 6),
            'min_samples_split': np.arange(4,5,1),
            # 'C': [10 ** i for i in range(-5, 6)],
        }

        return ret

    def predict(self, dataset, seed=123):
        executer = ThreadPoolExecutor(max_workers=self.__MAX_THREAD_COUNT)
        futures = []
        y_pred, y_true = [], []

        dataset_index = 0
        while dataset_index < len(dataset) or len(futures) > 0:
            if self.__used_thread_count < self.__MAX_THREAD_COUNT:
                clf = GridSearchCV(
                    RandomForestClassifier(
                        random_state=seed, 
                        class_weight='balanced'
                    ),
                    self.__param(),
                    cv=34,
                    scoring='f1_macro'
                )

        return y_pred, y_true