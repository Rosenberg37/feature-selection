# -*- coding: UTF-8 -*-
"""
@Project:   feature-selection 
@File:      utils.py
@Author:    Rosenberg
@Date:      2022/10/9 16:06 
@Documentation: 
    ...
"""
import os
from functools import reduce

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def get_csv_paths(data_path: str):
    csv_paths = list()
    for path in os.listdir(data_path):
        cur_path = os.path.join(data_path, path)
        if os.path.isdir(cur_path):
            csv_paths += get_csv_paths(cur_path)
        elif cur_path.endswith('.csv'):
            csv_paths.append(cur_path)
    return csv_paths


def deduplicate(a: list) -> iter:
    return reduce(lambda x, y: x if y in x else x + [y], [[], *a])


models = {
    '1NN': lambda: KNeighborsClassifier(1),
    '3NN': lambda: KNeighborsClassifier(3),
    '5NN': lambda: KNeighborsClassifier(5),
    'J48': lambda: DecisionTreeClassifier(criterion='entropy'),
    'NaiveBayes': lambda: GaussianNB(),
    'LogisticRegression': lambda: LogisticRegression(),
    'SVM': lambda: SVC(),
}

dataset_paths = {os.path.basename(csv_path)[:-4]: csv_path for csv_path in get_csv_paths(f'./data')}
