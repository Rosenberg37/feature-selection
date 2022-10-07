# -*- coding: UTF-8 -*-
"""
@Project:   feature-selection 
@File:      model.py
@Author:    Rosenberg
@Date:      2022/9/24 21:06 
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
from scoop import futures
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold


class FitnessFunction:
    def __init__(self, n_splits: int = 5):
        """

        :param n_splits: Number of splits for cv
        """
        self.n_splits: int = n_splits

    def __call__(self, model, data, feature_code):
        if np.sum(feature_code) == 0:
            return 0, 1

        feature_indices = np.where(feature_code == 1)[0]
        x, y = data[:, feature_indices], data[:, -1]

        cv_set = np.repeat(-1.0, x.shape[0])
        skf = StratifiedKFold(n_splits=self.n_splits)
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


class FeatureSelectionAlgo:
    """
    FeaturesSelectionGA
    This class uses Genetic Algorithm to find out the best features for an input model
    using Distributed Evolutionary Algorithms in Python(DEAP) package. Default toolbox is
    used for GA, but it can be changed accordingly.
    """

    def __init__(
            self,
            model=None,
            data=None,
            fitness_function: FitnessFunction = None
    ):
        """
        Parameters
        -----------
        model : scikit-learn supported model,
            x :  {array-like}, shape = [n_samples, n_features]
                 Training vectors, where n_samples is the number of samples
                 and n_features is the number of features.
            y  : {array-like}, shape = [n_samples]
                 Target Values
        """

        self.model = model
        self.data = data

        self.num_features = data.shape[1] - 1

        self.creator = self._default_creator()
        self.toolbox = self._default_toolbox()

        if fitness_function is None:
            self.fitness_function: FitnessFunction = FitnessFunction(n_splits=5)
        else:
            self.fitness_function: FitnessFunction = fitness_function

    @staticmethod
    def _default_creator():
        creator.create("FeatureSelObj", base.Fitness, weights=(1.0, 1.0))
        # 对于多目标优化问题，weights用来指示多个优化目标之间的相对重要程度以及最大化最小化。
        # (-1.0, -1.0)代表对第一个目标函数取最小值，对第二个目标函数取最小值。
        creator.create("Individual", list, fitness=creator.FeatureSelObj)
        return creator

    def _default_toolbox(self):
        toolbox = base.Toolbox()

        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            toolbox.attr_bool,
            self.num_features,
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.evaluate)
        toolbox.register("mate", tools.cxUniform, indpb=0.1)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)

        toolbox.register("map", futures.map)

        return toolbox

    def evaluate(self, individual):
        feature_code: np.ndarray = np.asarray(individual)
        return self.fitness_function(self.model, self.data, feature_code)

    def generate(
            self,
            num_population: int = 64,
            crossover_prob: float = 0.5,
            mutate_prob: float = 0.2,
            num_generation: int = 8,

    ):
        """
        Generate evolved population
        :param num_population: population size
        :param crossover_prob: crossover probability
        :param mutate_prob: mutation probability
        :param num_generation: number of generations
        :return: Fittest population
        """
        print("EVOLVING.......")
        pop = self.toolbox.population(num_population)

        # Evaluate the entire population
        fitness = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitness):
            ind.fitness.values = fit

        for g in range(num_generation):
            print(f"-- GENERATION {g + 1} --")

            # Select and clone the individuals
            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < crossover_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < mutate_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitness = list(map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitness):
                ind.fitness.values = fit
            print(f"Evaluated {len(invalid_ind)} individuals")

            # Environment select the elites
            pop = tools.selBest(offspring, num_population, fit_attr='fitness')

        print("-- Only the fittest survives --")
        best_individual = tools.selBest(pop, 1)[0]

        accuracy = best_individual.fitness.values[0]
        dim_reduction = 1 - sum(best_individual) / len(best_individual)

        print(f"Dimension reduction: {dim_reduction}.")
        print(f"Accuracy: {accuracy}.")

        return pop, accuracy, dim_reduction
