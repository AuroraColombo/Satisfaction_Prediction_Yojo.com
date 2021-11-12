import pandas as pd


def load_dataset(filepath, verbose=False):
    '''
    Load the dataset given a csv file. Also counts the identical values between Arrival delay in days and Shipping
    delay in days

    :param filepath: path to the file from the working directory
    :param verbose: True to print some dataset description
    :return: the dataset stored in a pandas.DataFrame
    '''
    df = pd.read_csv(filepath, index_col=0)
    if verbose is True:
        print(df.head(5))
        print(df.describe())
        print("There are ", end='')
        print((df['Arrival delay in days'] == df['Shipping delay in days']).sum(), end='')
        print(' identical values between Arrival delay in days and Shipping delay in days')
    return df


if __name__ == '__main__':
    load_dataset('../model.csv', True)
