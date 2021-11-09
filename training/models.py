from sklearn.tree import DecisionTreeClassifier


def get_decision_tree():
    tree = DecisionTreeClassifier()
    parameters = {'criterion': ['entropy', 'gini'],
                  'max_depth': [4, 5, 6, 8, 10],
                  'min_samples_split': [5, 10, 20],
                  'min_samples_leaf': [5, 10, 20]}

    return tree, parameters