from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from training.models import get_knn, get_multilayer_perceptron


def hyperp_search(classifier, parameters, dataframe, verbose=False):
    y = dataframe['Satisfied']
    X = dataframe.iloc[:, :-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=313)

    gs = GridSearchCV(classifier, parameters, cv=5, scoring='f1', verbose=10, n_jobs=-1)
    gs = gs.fit(X_train, y_train)

    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_train = best_model.predict(X_train)

    y_probs = best_model.predict_proba(X_test)
    y_probs_train = best_model.predict_proba(X_train)

    if verbose is True:
        print("f1_train: %f using %s" % (gs.best_score_, gs.best_params_))
        print("f1 score   train %.3f   test %.3f" % (f1_score(y_train, y_pred_train), f1_score(y_test, y_pred)))
        print("precision  train %.3f   test %.3f" % (
            precision_score(y_train, y_pred_train), precision_score(y_test, y_pred)))
        print("AUC        train %.3f   test %.3f" % (metrics.roc_auc_score(y_train, y_probs_train[:, 1]),
                                                     metrics.roc_auc_score(y_test, y_probs[:, 1])))
        print("")
        print(confusion_matrix(y_test, y_pred))


def test_model(dataframe):
    tree, params = get_knn()
    hyperp_search(tree, params, dataframe, True)
    # return results
