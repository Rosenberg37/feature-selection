# -*- coding: UTF-8 -*-
"""
@Project:   feature-selection 
@File:      algo.py
@Author:    Rosenberg
@Date:      2022/10/7 18:47 
@Documentation: 
    This file contains the implementation of the evolutionary algorithm for feature selection.
    Mainly reference to the following article:
        "https://www.jianshu.com/p/8fa044ed9267"
        "https://www.jianshu.com/p/3cbf5df95597"
        "https://www.jianshu.com/p/4873e16fa05a"
        "https://www.jianshu.com/p/a15d06645767"
        "https://www.jianshu.com/p/8e16fe258337"
        "https://www.jianshu.com/p/0b9da31f9ba3"
"""
import random

import numpy as np
import pandas as pd
from deap import base, creator, tools
from deap.tools.emo import assignCrowdingDist
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import utils

creator.create("FeatureSelObj", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", list, fitness=creator.FeatureSelObj)


def get_toolbox(
        model_name: str,
        dataset_name: str,
        pool=None
):
    toolbox = base.Toolbox()
    toolbox.dataset_name = dataset_name
    toolbox.model_name = model_name

    dataset_path = utils.dataset_paths[dataset_name]
    dataset = pd.read_csv(dataset_path).values

    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_bool,
        dataset.shape[1] - 1,
    )

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate, model_name=model_name, dataset=dataset)

    toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", tools.cxUniform, indpb=0.3)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.3)

    if pool is not None:
        toolbox.register("map", pool.map)

    return toolbox


def evaluate(
        individual: np.ndarray,
        model_name: str,
        dataset: np.ndarray,
        n_splits: int = 10
):
    feature_code: np.ndarray = np.asarray(individual)
    feature_indices = np.where(feature_code == 1)[0]
    if len(feature_indices) == 0:
        return 0, 0

    model = utils.models[model_name]()
    x, y = dataset[:, feature_indices], dataset[:, -1]

    cv_set = np.repeat(-1.0, x.shape[0])
    skf = StratifiedKFold(n_splits=n_splits)

    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(x_train, y_train)
        predicted_y = model.predict(x_test)
        cv_set[test_index] = predicted_y

    accuracy = accuracy_score(y, cv_set)
    deduction_rate = np.sum(feature_code == 0) / len(feature_code)
    return accuracy, deduction_rate


def feature_selection_with_nsga2(
        toolbox: base.Toolbox,
        num_generation: int = 100,
        num_population: int = 200,
        crossover_prob: float = 0.7,
        mutate_prob: float = 0.2,

):
    """
    Uses Genetic Algorithm to find out the best features for an input model
    using Distributed Evolutionary Algorithms in Python(DEAP) package. Default toolbox is
    used for GA, but it can be changed accordingly.
    :param toolbox: toolbox for the algorithm
    :param num_population: population size
    :param crossover_prob: crossover probability
    :param mutate_prob: mutation probability
    :param num_generation: number of generations
    :return: Fittest population
    """
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    print(f"Evolving on {toolbox.dataset_name} with {toolbox.model_name}.")

    # Generate population
    pop = toolbox.population(num_population)

    # Initialize attributes
    fitness = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitness):
        ind.fitness.values = fit
    assignCrowdingDist(pop)

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(pop), **record)

    for _ in tqdm(
            range(num_generation),
            bar_format='Generation {n}/{total}:|{bar}|[{elapsed}<{remaining},{rate_fmt}{postfix}]'
    ):
        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= crossover_prob:
                toolbox.mate(ind1, ind2)
                del ind1.fitness.values
                del ind2.fitness.values

        for ind in offspring:
            if random.random() <= mutate_prob:
                toolbox.mutate(ind)
                del ind.fitness.values

        pop = utils.deduplicate(pop + offspring)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitness = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitness):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.select(pop, num_population)

        record = stats.compile(pop)
        logbook.record(gen=0, evals=len(invalid_ind), **record)

    accuracy = pop[0].fitness.values[0]
    dim_reduction = pop[0].fitness.values[1]

    print(f"Dimension reduction: {dim_reduction}")
    print(f"Accuracy: {accuracy}\n")

    return pop, logbook
