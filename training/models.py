import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier


def get_decision_tree():
    tree = DecisionTreeClassifier()
    parameters = {'criterion': ['entropy', 'gini'],
                  'max_depth': np.arange(10, 20, 1),
                  'min_samples_split': np.arange(30, 51, 5),
                  'min_samples_leaf': np.arange(30, 60, 5)}

    return tree, parameters


def get_ensemble(base_estimator):
    bagging = BaggingClassifier(base_estimator=base_estimator)
    parameters = {'n_estimators': np.arange(2, 1000, 50),
                  'max_samples': np.arange(0.1, 1.1, 0.1),
                  'max_features': np.arange(0.1, 1.1, 0.1),
                  'bootstrap': [True, False]}
