from data.load_dataset import load_dataset
from data.preparation import *


def main():
    df = load_dataset('model.csv', True)
    df.drop(labels='id', axis=1, inplace=True)
    df = remove_duplicates(df, True)
    df, _ = remove_missing_values(df, True)
    df = categorical_to_dummy(df, True)

    # variables to log: price, shipping delay, arrival delay
    # StandardScaler only to numerical variables
    # PCA
