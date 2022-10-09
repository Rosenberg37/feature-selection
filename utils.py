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

from sklearn.linear_model import LogisticRegression, Perceptron
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


models = {
    'KNN': lambda: KNeighborsClassifier(5),
    'Perceptron': lambda: Perceptron(tol=1e-3, random_state=0),
    'DecisionTree': lambda: DecisionTreeClassifier(criterion='entropy'),
    'NaiveBayes': lambda: GaussianNB(),
    'LogisticRegression': lambda: LogisticRegression(),
    'SVM': lambda: SVC(),
}
dataset_paths = {os.path.basename(csv_path)[:-4]: csv_path for csv_path in get_csv_paths(f'./data')}
