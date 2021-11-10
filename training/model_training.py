from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, cross_val_score
from training.models import *
from tqdm import tqdm


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

    return best_model


def test_model(dataframe):
    model, params = get_knn()
    hyperp_search(model, params, dataframe, True)
    # return results


# def find_best_params(dataframe):
#     model, params = get_knn()
#
#     acc = []
#     x = []
#     start = 1
#     end = 10
#     length = range(start, end)
#
#     for i in tqdm(length):
#         knn = KNeighborsClassifier(n_neighbors=i)
#         _, c, _ = cross_val(knn, dataframe)
#         acc.append(c)
#         x.append(i)
#
#     # min samples split per ora inferiore a 315 ca
#     plt.plot(x, acc)
#     plt.grid()
#     # naming the x axis
#     plt.xlabel('Max_depth')
#     # naming the y axis
#     plt.ylabel('accuracy')
#
#     # giving a title to my graph
#
#     plt.title('Max_depth/Accuracy')
#
#     # plt.savefig("Min_impurity_decrease.png")
#     # function to show the plot
#     plt.show()
#
#     print(max(acc))
#     print(x[acc.index(max(acc))])
#
#
# def cross_val(model, dataframe):
#     y = dataframe['Satisfied']
#     X = dataframe.iloc[:, :-1]
#     cv = StratifiedKFold(n_splits=5, shuffle=False)
#     c = cross_val_score(model, X, y, cv=cv, scoring='f1')
#     return c, round(np.mean(c) * 100, 2), round(np.std(c) * 100, 2)
