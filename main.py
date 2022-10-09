import multiprocessing

from tabulate import tabulate

from model import feature_selection_with_nsga2, get_toolbox

if __name__ == '__main__':
    model_names = ['KNN', 'DecisionTree', 'SVM']
    dataset_names = ['vehicle', 'cleveland', 'heart', 'ionosphere', 'srbct', 'arcene']

    with multiprocessing.Pool() as pool:

        acc_dict, dr_dict = dict(), dict()
        for model_name in model_names:

            acc_list, dr_list = list(), list()
            for dataset_name in dataset_names:
                toolbox = get_toolbox(model_name, dataset_name, pool)
                _, acc, dr = feature_selection_with_nsga2(toolbox)

                acc_list.append(acc)
                dr_list.append(dr)

            acc_dict[model_name] = acc_list
            dr_dict[model_name] = dr_list

    print("----------|Accuracy Table|----------")
    print(tabulate({'Datasets': dataset_names, **acc_dict}, tablefmt='fancy_grid', headers='keys'))
    print("----------|Dimension Deduction Table|----------")
    print(tabulate({'Datasets': dataset_names, **dr_dict}, tablefmt='fancy_grid', headers='keys'))
