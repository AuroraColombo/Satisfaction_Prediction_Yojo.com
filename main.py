import matplotlib.pyplot as plt
import pandas as pd

from data.load_dataset import load_dataset
from data.preparation import *

def main():
    df = load_dataset('model.csv', True)
    df.drop(labels='id', axis=1, inplace=True)
    df = remove_duplicates(df, True)
    df, _ = remove_missing_values(df, True)
    df = categorical_to_dummy(df, True)

    # Normalizing variable 'Price' through log10 transformation

    df = feature_2_log(df, 'Price',  10) # Non applicherei la trasformazione logaritmica agli shipping e arrival, perchè mi sembra molto più normale senza (vedi hypernormal distribution)

    # StandardScaler only to numerical variables

    df = standardize(df)

    # PCA

    pca(df)

if __name__ == '__main__':
    main()