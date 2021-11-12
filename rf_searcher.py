import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from tqdm import tqdm

from data.load_dataset import load_dataset
from data.preparation import pca, standardize, feature_2_log, categorical_to_dummy, remove_duplicates, \
    remove_missing_values


def RandomForest(df, n_splits=10, n_estimators=10, criterion="gini", max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0., max_features=1.0,
                 random_state=0, max_leaf_nodes=None, min_impurity_decrease=0., max_samples=1.0,
                 class_weight=None, ccp_alpha=0.0, bootstrap=False):
    y = df['Satisfied']
    X = df.iloc[:, :-1]

    Random_Forest = RandomForestClassifier(n_estimators=n_estimators,
                                           criterion=criterion, max_depth=max_depth,
                                           min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                           min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                           random_state=random_state, max_leaf_nodes=max_leaf_nodes,
                                           min_impurity_decrease=min_impurity_decrease, class_weight=class_weight,
                                           ccp_alpha=ccp_alpha, bootstrap=bootstrap, max_samples=max_samples)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=False)
    c = cross_val_score(Random_Forest, X, y, cv=cv, scoring='f1', n_jobs=-1)

    return c, round(np.mean(c) * 100, 2), round(np.std(c) * 100, 2)


if __name__ == '__main__':

    dataframe = load_dataset('model.csv', False)
    dataframe.drop(labels=['id', 'Shipping delay in days'], axis=1, inplace=True)
    dataframe = remove_duplicates(dataframe, False)
    dataframe, df_wo_nan = remove_missing_values(dataframe, False)

    dataframe01 = dataframe.copy(deep=True)
    dataframe01.drop(labels=['Manufacturer sustainability', 'NewUsed', 'Category'], axis=1, inplace=True)
    dataframe19 = dataframe01.copy(deep=True)
    dataframe19.drop(labels=['Age', 'Product description accuracy', 'Arrival delay in days', 'Gender'], axis=1,
                     inplace=True)

    dataframe = categorical_to_dummy(dataframe, variables2convert=['Gender', 'Customer Type', 'NewUsed', 'Category',
                                                                   'Satisfaction'], verbose=False)
    df_wo_nan = categorical_to_dummy(df_wo_nan, variables2convert=['Gender', 'Customer Type', 'NewUsed', 'Category',
                                                                   'Satisfaction'], verbose=False)

    dataframe01 = categorical_to_dummy(dataframe01, variables2convert=['Gender', 'Customer Type', 'Satisfaction'],
                                       verbose=False)

    dataframe19 = categorical_to_dummy(dataframe19, variables2convert=['Customer Type', 'Satisfaction'], verbose=False)

    # Normalizing variable 'Price' through log10 transformation

    dataframe = feature_2_log(dataframe, 'Price', 10)
    df_wo_nan = feature_2_log(df_wo_nan, 'Price', 10)
    dataframe01 = feature_2_log(dataframe01, 'Price', 10)
    dataframe19 = feature_2_log(dataframe19, 'Price', 10)

    # StandardScaler only to numerical variables
    dataframe, scaler = standardize(dataframe, ['Age', 'Price', 'Arrival delay in days',
                                                'Product description accuracy', 'Manufacturer sustainability',
                                                'Packaging quality',
                                                'Additional options', 'Helpfulness of reviews and ratings',
                                                'Integrity of packaging',
                                                'Ease check-out procedure', 'Relevance of related products',
                                                'Costumer insurance'], False)

    df_wo_nan, scaler = standardize(df_wo_nan, ['Age', 'Price', 'Arrival delay in days',
                                                'Product description accuracy', 'Manufacturer sustainability',
                                                'Packaging quality',
                                                'Additional options', 'Helpfulness of reviews and ratings',
                                                'Integrity of packaging',
                                                'Ease check-out procedure', 'Relevance of related products',
                                                'Costumer insurance'], False)

    dataframe01, scaler = standardize(dataframe01, ['Age', 'Price', 'Arrival delay in days',
                                                    'Product description accuracy',
                                                    'Packaging quality',
                                                    'Additional options', 'Helpfulness of reviews and ratings',
                                                    'Integrity of packaging',
                                                    'Ease check-out procedure', 'Relevance of related products',
                                                    'Costumer insurance'], False)

    dataframe19, scaler = standardize(dataframe19, ['Price',
                                                    'Packaging quality',
                                                    'Additional options', 'Helpfulness of reviews and ratings',
                                                    'Integrity of packaging',
                                                    'Ease check-out procedure', 'Relevance of related products',
                                                    'Costumer insurance'], False)

    # Paired plot
    # paired_plot(dataframe, 'Satisfied')
    # plt.show()

    # PCA
    pca2, dataframe_pca = pca(dataframe, verbose=False)  # At this point dataframe is only scaled
    pca2, dataframe_pca01 = pca(dataframe01, verbose=False)  # At this point dataframe is only scaled
    pca2, dataframe_pca19 = pca(dataframe19, verbose=False)  # At this point dataframe is only scaled

    f1 = []
    x = []

    start = 1
    end = 2
    length = range(start, end)

    for i in tqdm(length):
        _, c, _ = RandomForest(df_wo_nan, n_splits=5, n_estimators=37, criterion='gini',
                               max_depth=29, random_state=0, min_samples_split=38,
                               min_samples_leaf=1, max_leaf_nodes=None, max_samples=0.1, max_features=0.49,
                               min_impurity_decrease=0.0, ccp_alpha=0.0, class_weight='balanced', bootstrap=False)

        f1.append(c)
        x.append(i)

    # min samples split per ora inferiore a 315 ca
    plt.plot(x, f1)
    plt.grid()
    # naming the x axis
    plt.xlabel('params')
    # naming the y axis
    plt.ylabel('F1')

    # giving a title to my graph

    plt.title('Param/F1')

    # plt.savefig("Min_impurity_decrease.png")
    # function to show the plot
    plt.show()

    print(max(f1))
    print(x[f1.index(max(f1))])
