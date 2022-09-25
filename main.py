import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from models.genetic_algorithm import FeatureSelectionGA, FitnessFunction

if __name__ == '__main__':
    model = KNeighborsClassifier(5)
    data = pd.read_csv('data/low/vehicle.csv').values
    X, y = data[:, :-1], data[:, -1]
    selector = FeatureSelectionGA(model, X, y, ff_obj=FitnessFunction())
    pop = selector.generate()
