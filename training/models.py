import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


def get_decision_tree():
    tree = DecisionTreeClassifier()
    parameters = {'criterion': ['entropy', 'gini'],
                  'max_depth': [4, 5, 6, 8, 10],
                  'min_samples_split': [5, 10, 20],
                  'min_samples_leaf': [5, 10, 20]}

    return tree, parameters

def get_knn():
    knn = KNeighborsClassifier()
    parameters = {'n_neighbors': np.arange(1, 1000)}

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