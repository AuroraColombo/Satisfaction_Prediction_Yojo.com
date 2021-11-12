import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class HeterogeneousEnsemble:
    """
    Class to build and use an heterogeneous ensemble model

    Attributes
    ----------
    mlp: Multilayer Perceptron
    rf: Random Forest
    dt: Decision Tree
    svm: Support Vector Machine

    Methods
    -------

    fit: fit the models given a dataset
    predict: make prediction on a given dataset
    test: perform a cross validation test to evaluate performance
    save_model: save the model to a pickle file
    @staticmethod load_model_from_filename: load a model from a pickle file
    """
    def __init__(self, threshold=2):
        """
        Initialize the estimators of the model

        :param threshold: the majority voting threshold to classify a record as 'Satisfied'
        """
        self.mlp = MLPClassifier(activation='tanh', alpha=0.0001, batch_size=250, hidden_layer_sizes=14,
                                 learning_rate='constant', max_iter=1000, solver='adam', tol=0.001, random_state=313)
        self.rf = RandomForestClassifier(n_estimators=37, criterion='gini',
                                         max_depth=29, random_state=313, min_samples_split=38,
                                         min_samples_leaf=1, max_leaf_nodes=None, max_samples=0.1, max_features=0.49,
                                         min_impurity_decrease=0.0, ccp_alpha=0.0, class_weight='balanced',
                                         bootstrap=False)
        self.dt = DecisionTreeClassifier(class_weight='balanced', criterion='gini', max_depth=15, min_samples_leaf=7,
                                         min_samples_split=50, random_state=313)

        self.svm = SVC(C=0.1, degree=3, gamma=0.5, kernel='poly', random_state=313)

        self.threshold = threshold

    def predict(self, X):
        """
        Make a prediction on a dataset

        :param X: the dataset to make the predictions on
        :return: the predicted values
        """
        dt_pred = self.dt.predict(X)
        rf_pred = self.rf.predict(X)
        mlp_pred = self.mlp.predict(X)
        svm_pred = self.svm.predict(X)

        results = self.__voting(dt_pred, rf_pred, mlp_pred, svm_pred)
        return results

    def fit(self, X, y):
        """
        Fit the estimators on the dataset X based on the target values in y

        :param X: the training input samples
        :param y: the target values as integers
        """
        self.dt.fit(X, y)
        self.rf.fit(X, y)
        self.mlp.fit(X, y)
        self.svm.fit(X, y)

    def __voting(self, tree_pred, forest_pred, ann_pred, svm_pred):
        """
        Perform the majority voting between the estimators

        :param tree_pred: the prediction of the decision tree
        :param forest_pred: the prediction of the random forest
        :param ann_pred: the prediction of the multilayer perceptron
        :param svm_pred: the prediction of the support vector machine
        :return: the predicted values after the voting
        """

        results = np.zeros(shape=(len(tree_pred),))

        for i in range(len(tree_pred)):
            count = 0
            if ann_pred[i] == 1:
                count += 1
            if tree_pred[i] == 1:
                count += 1
            if forest_pred[i] == 1:
                count += 1
            if svm_pred[i] == 1:
                count += 1
            if count >= self.threshold:
                results[i] = 1

        return results

    def cross_validation(self, dataframe):
        """
        Perform a cross validation evaluation on a given dataset

        :param dataframe: the dataset to perform cross validation on
        """
        y = dataframe['Satisfied'].values
        X = dataframe.iloc[:, :-1].values
        cv = StratifiedKFold(n_splits=5, shuffle=False)
        c = []

        X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X, y, test_size=0.2, stratify=y,
                                                                                    random_state=123)
        for train_index, test_index in cv.split(X_train_split, y_train_split):
            X_train, X_test = X_train_split[train_index], X_train_split[test_index]
            y_train, y_test = y_train_split[train_index], y_train_split[test_index]
            self.fit(X_train, y_train)
            y_pred = self.predict(X_test)
            c.append(f1_score(y_test, y_pred))

        print('Cross validation (mean std): ', end='')
        print(round(np.mean(c) * 100, 2), round(np.std(c) * 100, 2))

        y_pred = self.predict(X_test_split)
        y_pred_train = self.predict(X_train_split)
        print("f1  train %.3f   test %.3f" % (f1_score(y_train_split, y_pred_train), f1_score(y_test_split, y_pred)))

    def save_model(self):
        """
        Saves the model to a .pkl file
        """
        pickle.dump(self, open('heterogeneous_ensemble.pkl', 'wb'))

    @staticmethod
    def load_model_from_filename(filename):
        """
        Load a model from a .pkl file

        :param filename: the path to the .pkl file
        :return: an HeterogeneousEnsemble instance
        """
        return pickle.load(open(filename, 'rb'))
