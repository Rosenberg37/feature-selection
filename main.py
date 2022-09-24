from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

from models.genetic_algorithm import FeatureSelectionGA, FitnessFunction

if __name__ == '__main__':
    model = RandomForestClassifier(n_jobs=-1, n_estimators=5)
    X, y = load_breast_cancer(return_X_y=True)
    selector = FeatureSelectionGA(model, X, y, ff_obj=FitnessFunction())
    pop = selector.generate(10)
