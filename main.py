import os

import pandas as pd
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate

from model import FeatureSelectionNSGA


def get_csv_paths(data_path: str):
    csv_paths = list()
    for path in os.listdir(data_path):
        cur_path = os.path.join(data_path, path)
        if os.path.isdir(cur_path):
            csv_paths += get_csv_paths(cur_path)
        elif cur_path.endswith('.csv'):
            csv_paths.append(cur_path)
    return csv_paths


if __name__ == '__main__':
    models = {
        'KNN': KNeighborsClassifier(5),
        'Perceptron': Perceptron(tol=1e-3, random_state=0),
        'DecisionTree': DecisionTreeClassifier(),
        'NaiveBayes': GaussianNB(),
        'LogisticRegression': LogisticRegression()
    }

    csv_paths = get_csv_paths('./data')
    dataset_names = [os.path.basename(csv_path)[:-4] for csv_path in csv_paths]

    acc_dict, dr_dict = dict(), dict()
    for name, model in models.items():
        acc_list, dr_list = list(), list()
        for csv_path in csv_paths:
            data = pd.read_csv(csv_path).values

            selector = FeatureSelectionNSGA(model, data)
            pop, acc, dr = selector.generate()

            acc_list.append(acc)
            dr_list.append(dr)

        acc_dict[name] = acc_list
        dr_dict[name] = dr_list

    print("----------|Accuracy Table|----------")
    print(tabulate({'Datasets': dataset_names, **acc_dict}, tablefmt='fancy_grid', headers='keys'))
    print("----------|Dimension Deduction Table|----------")
    print(tabulate({'Datasets': dataset_names, **dr_dict}, tablefmt='fancy_grid', headers='keys'))
