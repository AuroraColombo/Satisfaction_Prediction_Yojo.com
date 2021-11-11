import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns


def remove_duplicates(dataframe, verbose=False):
    duplicates = dataframe.duplicated(keep='first')
    if verbose is True:
        print("There are %d duplicates" % sum(duplicates))

    dataframe.drop_duplicates(inplace=True)

    if verbose is True:
        print("The dataset contains %d records" % dataframe.shape[0])

    return dataframe


def remove_missing_values(dataframe, verbose=False):
    if verbose is True:
        print("There are %d missing values" % sum(dataframe.isna().sum()))
        print(dataframe.isna().sum())

    dataframe_wo_nan = dataframe.dropna(axis=0, how='any', inplace=False)
    dataframe['Age'] = dataframe['Age'].fillna(-1)

    if verbose is True:
        print("The dataset without missing values contains %d records" % dataframe_wo_nan.shape[0])

    return dataframe, dataframe_wo_nan


def categorical_to_dummy(dataframe, variables2convert, verbose=False):
    for variable in variables2convert:
        dummy = pd.get_dummies(dataframe[variable], drop_first=True)
        dataframe = dataframe.drop([variable], axis=1, inplace=False)
        dataframe = pd.concat([dataframe, dummy], axis=1)

    if verbose is True:
        print(dataframe.head(5))

    return dataframe


def standardize(dataframe, features, verbose=False):
    scaler = StandardScaler()
    dataframe_stand = dataframe.copy()
    scaler.fit(dataframe_stand.loc[:, :].astype(float))  # provo a selezionare qui i parametri
    dataframe_stand = pd.DataFrame(scaler.transform(dataframe_stand.loc[:, :].astype(float)))
    dataframe_stand.columns = dataframe.columns

    dataframe[features] = dataframe_stand[features]

    if verbose is True:
        dataframe_stand.hist()
        plt.show()
        print(dataframe.head(5))

    return dataframe, scaler


def feature_2_log(dataframe, feature, log_base):
    if min(dataframe.loc[:, feature]) < 0:
        offset = math.ceil(abs(min(dataframe.loc[:, feature])))
    else:
        offset = 1
    dataframe.loc[:, feature] = dataframe[feature].apply(lambda x: math.log(x + offset, log_base))

    return dataframe


def pca(dataframe, verbose=False):
    pca2 = PCA()

    dataframe_pca = dataframe.copy()
    dataframe_pca.drop(labels='Satisfied', axis=1, inplace=True)

    pca2.fit(dataframe_pca)
    dataframe_pca = pd.DataFrame(pca2.transform(dataframe_pca))
    dataframe_pca['Satisfied'] = dataframe['Satisfied']

    if verbose is True:
        print("\n\n PCA: \n")
        print("Dataset shape before PCA: ", str(dataframe.shape) + "\n")
        print("Dataset shape after PCA: ", str(dataframe_pca.shape) + "\n")

        print("Attributes variance:" + str(pd.DataFrame(pca2.explained_variance_).transpose()) + "\n")

        explained_var = pd.DataFrame(pca2.explained_variance_ratio_).transpose()
        sns.barplot(data=explained_var)
        plt.show()

        cum_explained_var = np.cumsum(pca2.explained_variance_ratio_)
        plt.plot(cum_explained_var)
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.show()

    return pca2, dataframe_pca


def paired_plot(dataframe, target):
    sns.pairplot(dataframe, hue=target)
    plt.show()
