import multiprocessing

from tabulate import tabulate

from model import feature_selection_with_nsga2, get_toolbox
from utils import sort_population

# baseline and setting from "http://kns.cnki.net/kcms/detail/11.2560.TP.20171113.1413.002.html"
experiment_settings = [
    ['5NN', 'ionosphere', 10, 0.9886, 0.8765],
    ['3NN', 'ionosphere', 10, 0.9942, 0.8735],
    ['1NN', 'ionosphere', 0.3, 0.9685, 0.7911],
    ['SVM', 'ionosphere', 2, 0.9573, 0.7058],
    ['J48', 'ionosphere', 0.3, 0.9781, 0.6029],
    ['J48', 'ionosphere', 10, 0.9914, 0.7823],
    ['1NN', 'cleveland', 0.3, 0.5977, 0.6153],
    ['1NN', 'srbct', 0.3, 0.9555, 0.8966],
    ['5NN', 'vehicle', 0.3, 0.7539, 0.50],
    ['SVM', 'vehicle', 2, 0.6962, 0.75],
    ['SVM', 'vehicle', 10, 0.8357, 0.4222],
    ['3NN', 'heart', 10, 0.9185, 0.70],
    ['SVM', 'heart', 2, 0.8444, 0.5739],
    ['J48', 'heart', 10, 0.9370, 0.6153],
    ['J48', 'arcene', 0.3, 0.7685, 0.954]
]

if __name__ == '__main__':
    with multiprocessing.Pool() as pool:
        for setting in experiment_settings:
            toolbox = get_toolbox(*setting[:3], pool=pool)
            gens, _ = feature_selection_with_nsga2(toolbox)
            pop = sort_population(gens[-1])

            setting += [*pop[0].fitness.values]

    experiments = {
        'Model': list(map(lambda x: x[0], experiment_settings)),
        'Dataset': list(map(lambda x: x[1], experiment_settings)),
        'Setting': list(map(lambda x: x[2], experiment_settings)),
        'Baseline Acc': list(map(lambda x: x[3], experiment_settings)),
        'Baseline DR': list(map(lambda x: x[4], experiment_settings)),
        'Acc': list(map(lambda x: x[5], experiment_settings)),
        'DR': list(map(lambda x: x[6], experiment_settings)),
        'Acc Compare': list(map(lambda x: '+' if x[5] > x[3] else '-', experiment_settings)),
        'DR Compare': list(map(lambda x: '+' if x[6] > x[4] else '-', experiment_settings)),
    }
    print("----------|Experiment Result|----------")
    print(tabulate(experiments, tablefmt='fancy_grid', headers='keys'))
