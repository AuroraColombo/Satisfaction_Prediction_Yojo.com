import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier


def get_decision_tree():
    tree = DecisionTreeClassifier()
    parameters = {'criterion': ['entropy', 'gini'],
                  'max_depth': np.arange(10, 20, 1),
                  'min_samples_split': np.arange(30, 51, 5),
                  'min_samples_leaf': np.arange(30, 60, 5)}

    return tree, parameters


def get_knn():
    knn = KNeighborsClassifier()
    parameters = {'n_neighbors': [5, 10, 40]}

    return knn, parameters


def get_naive_bayes():
    model = GaussianNB()
    parameters = {}

    return model, parameters


def get_multilayer_perceptron():
    nn = MLPClassifier()
    parameters = {'hidden_layer_sizes': np.arange(2, 10, 2),
                  'activation': ['relu', 'sigmoid', 'tanh'],
                  'alpha': [0.0001],
                  'batch_size': np.arange(200, 1000, 200),
                  'learning_rate': ['constant', 'lbfgs', 'invscaling', 'adaptive'],
                  'max_iter': np.arange(250, 1000, 250),
                  'solver': ['sgd', 'adam'],
                  'tol': [0.01, 0.001, 0.0001]}

    return nn, parameters


def get_ensemble(base_estimator, base_etimator_parameters):
    bagging = BaggingClassifier(base_estimator=base_estimator)
    parameters = {'n_estimators': np.arange(2, 1000, 50),
                  'max_samples': np.arange(0.1, 1.1, 0.1),
                  'max_features': np.arange(0.1, 1.1, 0.1),
                  'bootstrap': [True, False]}


