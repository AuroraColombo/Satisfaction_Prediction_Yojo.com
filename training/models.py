import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
from sklearn import metrics


def get_decision_tree(): # Raffa
    tree = DecisionTreeClassifier()
    parameters = {'criterion': ['entropy', 'gini'],
                  'max_depth': np.arange(10, 20, 1),
                  'min_samples_split': np.arange(30, 51, 5),
                  'min_samples_leaf': np.arange(30, 60, 5)}

    return tree, parameters


# def get_knn(): # Au
#    knn = KNeighborsClassifier()
#    parameters = {'n_neighbors': np.arange(5, 10, 1)}

    return knn, parameters


def get_naive_bayes():
    model = GaussianNB()
    parameters = {}

    return model, parameters


def get_logistic_regression(): # Davi
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
                  "gamma": np.arange(0.5, 1.5, 0.01),
                  "degree": [2, 3, 4]}

    return classifier, parameters


def test_knn(dataframe, start, stop, step):
    score_train = []
    score_test = []

    x = dataframe.iloc[:, :-1]
    y = dataframe.iloc[:, -1]  # [-1]]

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.30,  # by default is 75%-25%
                                                        # shuffle is set True by default,
                                                        stratify=y,
                                                        random_state=123
                                                        )  # fix random seed for replicability

    neighbors = range(start, stop, step)

    for i in neighbors:
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(x_train, y_train)
        y_pred_train = knn.predict(x_train)
        y_pred_test = knn.predict(x_test)
        score_train.append(metrics.f1_score(y_train, y_pred_train))
        score_test.append(metrics.f1_score(y_test, y_pred_test))

    plt.xlabel('Neighbors')
    plt.ylabel('F1')
    plt.plot(neighbors,score_train, color='blue', alpha=1.00)
    plt.plot(neighbors,score_test, color='red', alpha=1.00)

    plt.show()

