import math

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
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


def categorical_to_dummy(dataframe, verbose=False):
    variables2convert = ['Gender', 'Customer Type', 'NewUsed', 'Category', 'Satisfaction']

    for variable in variables2convert:
        dummy = pd.get_dummies(dataframe[variable], drop_first=True)
        dataframe = dataframe.drop([variable], axis=1, inplace=False)
        dataframe = pd.concat([dataframe, dummy], axis=1)

    if verbose is True:
        print(dataframe.head(5))

    return dataframe


def standardize(dataframe):
    scaler = StandardScaler()
    scaler.fit(dataframe.loc[:, :].astype(float))  # provo a selezionare qui i parametri
    dataframe_stand = pd.DataFrame(scaler.transform(dataframe.loc[:, :].astype(float)))
    dataframe_stand.columns = dataframe.columns
    dataframe_stand.hist()
    plt.show()

    return dataframe_stand, scaler


def feature_2_log(dataframe, feature, log_base):
    if min(dataframe.loc[:, feature]) < 0:
        offset = math.ceil(abs(min(dataframe.loc[:, feature])))
    else:
        offset = 1
    dataframe.loc[:, feature] = dataframe[feature].apply(lambda x: math.log(x + offset, log_base))

    return dataframe


def pca(dataframe):
    pca2 = PCA()
    pca2.fit(dataframe) # il problema è qui dentro :)
    dataframe_pca = pd.DataFrame(pca2.transform(dataframe))

    print("\n\n PCA: \n")
    print("Dataset shape before PCA: ", dataframe.shape + "\n")
    print("Dataset shape after PCA: ", dataframe_pca.shape + "\n")

    print("Attributes variance:" + pd.DataFrame(pca.explained_variance_).transpose() + "\n")

    explained_var = pd.DataFrame(pca2.explained_variance_ratio_).transpose()
    sns.barplot(data=explained_var)
    plt.show()