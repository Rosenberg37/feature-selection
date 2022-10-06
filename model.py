# -*- coding: UTF-8 -*-
"""
@Project:   feature-selection 
@File:      genetic_algorithm.py
@Author:    Rosenberg
@Date:      2022/9/24 21:06 
@Documentation: 
    This file contains the implementation of the genetic algorithm for feature selection.
"""
import random

import numpy as np
from deap import base, creator, tools
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold


class FeatureSelectionGA:
    """
    FeaturesSelectionGA
    This class uses Genetic Algorithm to find out the best features for an input model
    using Distributed Evolutionary Algorithms in Python(DEAP) package. Default toolbox is
    used for GA, but it can be changed accordingly.
    """

    def __init__(
            self,
            model=None,
            x=None,
            y=None,
            fitness_function=None
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
        self.x = x
        self.y = y

        self.n_features = x.shape[1]

        self.creator = self._default_creator()
        self.toolbox = self._default_toolbox()

        self.final_fitness = []
        self.fitness_in_generation = {}
        self.best_individual = None

        if fitness_function is None:
            self.fitness_function = FitnessFunction(n_splits=5)
        else:
            self.fitness_function = fitness_function

    @staticmethod
    def _default_creator():
        creator.create("FeatureSelect", base.Fitness, weights=(1.0,))
        # 在创建单目标优化问题时，weights用来指示最大化和最小化。
        # -1.0即代表问题是一个最小化问题，对于最大化，应将weights改为正数，如1.0。
        creator.create("Individual", list, fitness=creator.FeatureSelect)
        return creator

    def _default_toolbox(self):
        toolbox = base.Toolbox()
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            toolbox.attr_bool,
            self.n_features,
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", self.evaluate)
        return toolbox

    def evaluate(self, individual):
        np_ind = np.asarray(individual)

        if np.sum(np_ind) == 0:
            fitness = 0.0
        else:
            feature_idx = np.where(np_ind == 1)[0]
            fitness = self.fitness_function(self.model, self.x[:, feature_idx], self.y)

        return fitness,

    def generate(
            self,
            num_population: int = 8,
            crossover_prob: float = 0.5,
            mutate_prob: float = 0.2,
            num_generation: int = 4,

    ):
        """
        Generate evolved population
        :param num_population: population size
        :param crossover_prob: crossover probability
        :param mutate_prob: mutation probability
        :param num_generation: number of generations
        :return: Fittest population
        """
        pop = self.toolbox.population(num_population)

        # Evaluate the entire population
        print("EVOLVING.......")
        fitness = list(map(self.toolbox.evaluate, pop))

        for ind, fit in zip(pop, fitness):
            ind.fitness.values = fit

        for g in range(num_generation):
            print(f"-- GENERATION {g + 1} --")
            offspring = self.toolbox.select(pop, len(pop))
            self.fitness_in_generation[str(g + 1)] = max(
                [ind.fitness.values[0] for ind in pop]
            )
            # Clone the selected individuals
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
            weak_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitness = list(map(self.toolbox.evaluate, weak_ind))
            for ind, fit in zip(weak_ind, fitness):
                ind.fitness.values = fit
            print(f"Evaluated {len(weak_ind)} individuals")

            # The population is entirely replaced by the offspring
            pop[:] = offspring

            # Gather all the fitness in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        print("-- Only the fittest survives --")
        self.best_individual = tools.selBest(pop, 1)[0]

        accuracy = self.best_individual.fitness.values[0]
        dim_reduction = 1 - sum(self.best_individual) / len(self.best_individual)

        print(f"Dimension reduction: {dim_reduction}.")
        print(f"Accuracy: {accuracy}.")

        self.final_fitness = list(zip(pop, fits))
        return pop, accuracy, dim_reduction


class FitnessFunction:
    def __init__(self, n_splits: int = 5):
        """

        :param n_splits: Number of splits for cv
        """
        self.n_splits: int = n_splits

    def __call__(self, model, x, y):
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
        return accuracy_score(y, cv_set)
