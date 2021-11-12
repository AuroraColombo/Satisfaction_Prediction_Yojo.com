import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class HeterogeneousEnsemble:
    def __init__(self, threshold=2):
        self.mlp = MLPClassifier(activation='tanh', alpha=0.0001, batch_size=250, hidden_layer_sizes=14,
                                 learning_rate='constant', max_iter=1000, solver='adam', tol=0.001)
        self.rf = RandomForestClassifier(n_estimators=37, criterion='gini',
                                         max_depth=29, random_state=0, min_samples_split=38,
                                         min_samples_leaf=1, max_leaf_nodes=None, max_samples=0.1, max_features=0.49,
                                         min_impurity_decrease=0.0, ccp_alpha=0.0, class_weight='balanced',
                                         bootstrap=False)
        self.dt = DecisionTreeClassifier(class_weight='balanced', criterion='gini', max_depth=15, min_samples_leaf=7,
                                         min_samples_split=50)

        # self.dt = DecisionTreeClassifier(class_weight='balanced', criterion='gini', max_depth=14, min_samples_leaf=12,
        #                                 min_samples_split=59)

        # self.svm = SVC(C=0.001, degree=3, gamma=1.0, kernel='poly')

        self.svm = SVC(C=0.1, degree=3, gamma=0.5, kernel='poly')

        # self.adaboost = AdaBoostClassifier(learning_rate=1.0, n_estimators=210)

        # self.lr = LogisticRegression(max_iter=400, penalty='none', solver='sag', tol=0.01)

        self.threshold = threshold

    def predict(self, X):
        dt_pred = self.dt.predict(X)
        rf_pred = self.rf.predict(X)
        mlp_pred = self.mlp.predict(X)
        svm_pred = self.svm.predict(X)
        # lr_pred = self.lr.predict(X)
        # ada_pred = self.adaboost.predict(X)

        results = self.__voting(dt_pred, rf_pred, mlp_pred, svm_pred)
        return results

    def fit(self, X, y):
        self.dt.fit(X, y)
        self.rf.fit(X, y)
        self.mlp.fit(X, y)
        self.svm.fit(X, y)
        # self.lr.fit(X, y)
        # self.adaboost.fit(X, y)

    def __voting(self, tree_pred, forest_pred, ann_pred, svm_pred):

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
            # if lr_pred[i] == 1:
            #     count += 1
            # if ada_pred[i] == 1:
            #     count += 1
            if count >= self.threshold:
                results[i] = 1

        return results

    @staticmethod
    def __load_sklearn_model(model_path):
        model = pickle.load(open(model_path, 'rb'))
        return model
