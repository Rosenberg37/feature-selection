# -*- coding: UTF-8 -*-
"""
@Project:   feature-selection 
@File:      genetic_algorithm.py
@Author:    Rosenberg
@Date:      2022/9/24 21:06 
@Documentation: 
    ...
"""
import random

import numpy as np
from deap import base, creator, tools
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold


class FitnessFunction:
    def __init__(self, n_splits=5):
        """
        Parameters
        -----------
        n_splits :int,
            Number of splits for cv
        """
        self.n_splits = n_splits

    def calculate_fitness(self, model, x, y):
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
        # return f1_score(y,cv_set)
        return accuracy_score(y, cv_set)


class FeatureSelectionGA:
    """
    FeaturesSelectionGA
    This class uses Genetic Algorithm to find out the best features for an input model
    using Distributed Evolutionary Algorithms in Python(DEAP) package. Default toolbox is
    used for GA, but it can be changed accordingly.
    """

    def __init__(self, model, x, y, verbose=0, ff_obj=None):
        """
        Parameters
        -----------
        model : scikit-learn supported model,
            x :  {array-like}, shape = [n_samples, n_features]
                 Training vectors, where n_samples is the number of samples
                 and n_features is the number of features.
            y  : {array-like}, shape = [n_samples]
                 Target Values
        verbose: 0 or 1
        """
        self.model = model
        self.n_features = x.shape[1]
        self.toolbox = None
        self.creator = self._create()
        # self.cv_split = cv_split
        self.x = x
        self.y = y
        self.verbose = verbose
        if self.verbose == 1:
            print(
                "Model {} will select best features among {} features.".format(
                    model, x.shape[1]
                )
            )
            print("Shape od train_x: {} and target: {}".format(x.shape, y.shape))
        self.final_fitness = []
        self.fitness_in_generation = {}
        self.best_ind = None
        if ff_obj is None:
            self.fitness_function = FitnessFunction(n_splits=5)
        else:
            self.fitness_function = ff_obj

    def evaluate(self, individual):
        fit_obj = self.fitness_function
        np_ind = np.asarray(individual)
        if np.sum(np_ind) == 0:
            fitness = 0.0
        else:
            feature_idx = np.where(np_ind == 1)[0]
            fitness = fit_obj.calculate_fitness(
                self.model, self.x[:, feature_idx], self.y
            )

        if self.verbose == 1:
            print("Individual: {}  Fitness_score: {} ".format(individual, fitness))

        return fitness,

    @staticmethod
    def _create():
        creator.create("FeatureSelect", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FeatureSelect)
        return creator

    def create_toolbox(self):
        """
        Custom creation of toolbox.
        Parameters
        -----------
            self
        Returns
        --------
            Initialized toolbox
        """

        return self._init_toolbox()

    def register_toolbox(self, toolbox):
        """
        Register custom created toolbox. Evaluate function will be registered
        in this method.
        Parameters
        -----------
            Registered toolbox with crossover,mutate,select tools except evaluate
        Returns
        --------
            self
        """
        toolbox.register("evaluate", self.evaluate)
        self.toolbox = toolbox

    def _init_toolbox(self):
        toolbox = base.Toolbox()
        toolbox.register("attr_bool", random.randint, 0, 1)
        # Structure initializers
        toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            toolbox.attr_bool,
            self.n_features,
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        return toolbox

    def _default_toolbox(self):
        toolbox = self._init_toolbox()
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", self.evaluate)
        return toolbox

    def get_final_scores(self, pop, fits):
        self.final_fitness = list(zip(pop, fits))

    def generate(self, num_population=64, crossover_prob=0.5, mutate_prob=0.2, num_generation=8, set_toolbox=False):
        """
        Generate evolved population
        Parameters
        -----------
            num_population : {int}
                    population size
            crossover_prob  : {float}
                    crossover probability
            mutate_prob: {float}
                    mutation probability
            num_generation : {int}
                    number of generations
            set_toolbox : {boolean}
                          If True then you have to create custom toolbox before calling
                          method. If False use default toolbox.
        Returns
        --------
            Fittest population
        """

        if self.verbose == 1:
            print(
                "Population: {}, crossover_probability: {}, mutation_probability: {}, total generations: {}".format(
                    num_population, crossover_prob, mutate_prob, num_generation
                )
            )

        if not set_toolbox:
            self.toolbox = self._default_toolbox()
        else:
            raise Exception(
                "Please create a toolbox.Use create_toolbox to create and register_toolbox to register. Else set set_toolbox = False to use default toolbox"
            )
        pop = self.toolbox.population(num_population)

        # Evaluate the entire population
        print("EVOLVING.......")
        fitness = list(map(self.toolbox.evaluate, pop))

        for ind, fit in zip(pop, fitness):
            ind.fitness.values = fit

        for g in range(num_generation):
            print("-- GENERATION {} --".format(g + 1))
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
            print("Evaluated %i individuals" % len(weak_ind))

            # The population is entirely replaced by the offspring
            pop[:] = offspring

            # Gather all the fitness in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5
        if self.verbose == 1:
            print("  Min %s" % min(fits))
            print("  Max %s" % max(fits))
            print("  Avg %s" % mean)
            print("  Std %s" % std)

        print("-- Only the fittest survives --")

        self.best_ind = tools.selBest(pop, 1)[0]
        print(f"Best individual is {self.best_ind}.\n")
        print(f"Dimension reduction is {1 - sum(self.best_ind) / len(self.best_ind)}.\n")
        print(f"Accuracy is  {self.best_ind.fitness.values[0]}.\n")

        self.get_final_scores(pop, fits)

        return pop
