import pandas as pd


def load_dataset(filepath, verbose=False):
    df = pd.read_csv(filepath, index_col=0)
    df.drop(labels='id', axis=1, inplace=True)
    if verbose is True:
        print(df.head(5))
        print(df.describe())
    return df


if __name__ == '__main__':
    load_dataset('../model.csv', True)
