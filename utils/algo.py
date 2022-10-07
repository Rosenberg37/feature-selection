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
from deap import base, creator, tools
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

creator.create("FeatureSelObj", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", list, fitness=creator.FeatureSelObj)


def get_toolbox(
        model,
        data,
        pool=None
):
    toolbox = base.Toolbox()

    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_bool,
        data.shape[1] - 1,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate, model=model, data=data)

    toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", tools.cxUniform, indpb=0.1)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)

    if pool is not None:
        toolbox.register("map", pool.map)

    return toolbox


def evaluate(
        individual,
        model,
        data,
        n_splits: int = 5
):
    feature_code: np.ndarray = np.asarray(individual)
    if np.sum(feature_code) == 0:
        return 0, 1

    feature_indices = np.where(feature_code == 1)[0]
    x, y = data[:, feature_indices], data[:, -1]

    cv_set = np.repeat(-1.0, x.shape[0])
    skf = StratifiedKFold(n_splits=n_splits)
    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if x_train.shape[0] != y_train.shape[0]:
            raise Exception()
        model.fit(x_train, y_train)
        predicted_y = model.predict(x_test)
        cv_set[test_index] = predicted_y

    accuracy = accuracy_score(y, cv_set)
    deduction_rate = np.sum(feature_code == 0) / len(feature_code)
    return accuracy, deduction_rate


def feature_selection_with_nsga(
        toolbox,
        num_generation: int = 64,
        num_population: int = 128,
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
    print("EVOLVING.......")
    pop = toolbox.population(num_population)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitness = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitness):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))

    for g in range(num_generation):
        print(f"-- GENERATION {g + 1} --")

        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= crossover_prob:
                toolbox.mate(ind1, ind2)

        for ind in offspring:
            if random.random() <= mutate_prob:
                toolbox.mutate(ind)
            del ind.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitness = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitness):
            ind.fitness.values = fit

        print(f"Evaluated {len(invalid_ind)} individuals")

        # Select the next generation population
        pop = toolbox.select(pop + offspring, num_population)

    print("-- Only the fittest survives --")
    best_individual = tools.selBest(pop, 1)[0]

    accuracy = best_individual.fitness.values[0]
    dim_reduction = best_individual.fitness.values[1]

    print(f"Dimension reduction: {dim_reduction}.")
    print(f"Accuracy: {accuracy}.")

    return pop, accuracy, dim_reduction
