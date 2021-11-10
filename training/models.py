import numpy as np
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier


# from sklearn.model_selection import GridSearchCV

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


def get_logistic_regression():
    model = LogisticRegression()
    parameters = {'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                  'tol': [0.01, 0.001],
                  'C': np.arange(5, 15, 1),
                  'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                  'max_iter': np.arange(250, 1000, 250)}

    return model, parameters


def get_multilayer_perceptron():
    nn = MLPClassifier()
    parameters = {'hidden_layer_sizes': np.arange(2, 10, 2),
                  'activation': ['relu', 'sigmoid', 'tanh'],
                  'alpha': [0.0001],
                  'batch_size': np.arange(250, 1000, 250),
                  'learning_rate': ['constant', 'lbfgs', 'invscaling', 'adaptive'],
                  'max_iter': np.arange(250, 1000, 250),
                  'solver': ['sgd', 'adam'],
                  'tol': [0.01, 0.001]}

    return nn, parameters


def get_ensemble(base_estimator):
    bagging = BaggingClassifier(base_estimator=base_estimator)
    parameters = {'n_estimators': np.arange(2, 1000, 50),
                  'max_samples': np.arange(0.1, 1.1, 0.1),
                  'max_features': np.arange(0.1, 1.1, 0.1),
                  'bootstrap': [True, False]}
    return bagging, parameters


def get_adaboost():
    adaboost = AdaBoostClassifier()
    parameters = {'n_estimators': [5, 10, 20, 30],
                  'learning_rate': [0.0001, 0.01, 0.1, 1, 10]}

    return adaboost, parameters


def get_svm():
    classifier = SVC()
    parameters = {"kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
                  "C": [0.1, 100],
                  "gamma": [1],
                  "degree": [2, 3, 4]}

    return classifier, parameters
