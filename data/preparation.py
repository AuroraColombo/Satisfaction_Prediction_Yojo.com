import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler


def remove_duplicates(dataframe, verbose=False):
    '''
    Removes the duplicate in a pandas.DataFrame

    :param dataframe: the dataset stored in a pandas.Dataframe object
    :param verbose: True to print some information on the execution
    :return: a pandas.DataFrame without duplicates samples
    '''
    duplicates = dataframe.duplicated(keep='first')
    if verbose is True:
        print("There are %d duplicates" % sum(duplicates))

    dataframe.drop_duplicates(inplace=True)

    if verbose is True:
        print("The dataset contains %d records" % dataframe.shape[0])

    return dataframe


def remove_missing_values(dataframe, verbose=False):
    '''
    Removes all the records with at least one missing values

    :param dataframe: the dataset stored in a pandas.Dataframe object
    :param verbose: True to print some information on the execution
    :return: a the original dataset and a dataset without samples containing missing values
    '''
    if verbose is True:
        print("There are %d missing values" % sum(dataframe.isna().sum()))
        print(dataframe.isna().sum())

    dataframe_wo_nan = dataframe.dropna(axis=0, how='any', inplace=False)
    dataframe_wo_nan.reset_index(drop=True, inplace=True)
    dataframe['Age'] = dataframe['Age'].fillna(-1)

    if verbose is True:
        print("The dataset without missing values contains %d records" % dataframe_wo_nan.shape[0])

    return dataframe, dataframe_wo_nan


def categorical_to_dummy(dataframe, variables2convert, verbose=False):
    '''
    Converts the selected attributes into dummy variables. Drops the last dummy variable for each attribute

    :param dataframe: the pandas.DataFrame with the attributes to convert
    :param variables2convert: a list containing the column names to convert
    :param verbose: True to print information on the execution
    :return: the dataset with the dummy variables converted
    '''
    for variable in variables2convert:
        dummy = pd.get_dummies(dataframe[variable], drop_first=True)
        dataframe = dataframe.drop([variable], axis=1, inplace=False)
        dataframe = pd.concat([dataframe, dummy], axis=1)

    if verbose is True:
        print(dataframe.head(5))

    return dataframe


def standardise(dataframe, features, verbose=False):
    """
    Applies the sklearn.preprocessing.StandardScaler to the features selected

    :param dataframe: the dataframe containing the variables to scale
    :param features: a list of all the attributes to be scaled
    :param verbose: True to print some information on the execution
    :return: the dataset with the converted attributes and the StandardScaler() fitted
    """
    scaler = StandardScaler()
    dataframe_stand = dataframe.copy()  # copy to keep the variables that should not be scaled
    scaler.fit(dataframe_stand.loc[:, :].astype(float))
    dataframe_stand = pd.DataFrame(scaler.transform(dataframe_stand.loc[:, :].astype(float)))
    dataframe_stand.columns = dataframe.columns

    dataframe[features] = dataframe_stand[features]

    if verbose is True:
        dataframe_stand.hist()
        plt.show()
        print(dataframe.head(5))

    return dataframe, scaler


def feature_2_log(dataframe, feature, log_base):
    """
    Apply a logarithmic function to a specific feature of a dataset

    :param dataframe: the dataset containing the feature to transform
    :param feature: the attribute to apply the log function to
    :param log_base: the base of the logarithmic function to apply
    :return: the dataset with the converted attribute
    """
    if min(dataframe.loc[:, feature]) < 0:  # offset to be added to the variable to avoid the log(0) issue
        offset = math.ceil(abs(min(dataframe.loc[:, feature])))
    else:
        offset = 1
    dataframe.loc[:, feature] = dataframe[feature].apply(lambda x: math.log(x + offset, log_base))

    return dataframe
