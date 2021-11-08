import pandas as pd


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

    if verbose is True:
        print("The dataset contains %d records" % dataframe_wo_nan.shape[0])

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
