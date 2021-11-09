import matplotlib.pyplot as plt
import pandas as pd

from data.load_dataset import load_dataset
from data.preparation import *
from training.model_training import test_model


def main():
    dataframe = load_dataset('model.csv', True)
    dataframe.drop(labels='id', axis=1, inplace=True)
    dataframe = remove_duplicates(dataframe, True)
    dataframe, df_wo_nan = remove_missing_values(dataframe, True)
    dataframe = categorical_to_dummy(dataframe, True)

    # Normalizing variable 'Price' through log10 transformation

    dataframe = feature_2_log(dataframe, 'Price', 10)
    # StandardScaler only to numerical variables

    dataframe, scaler = standardize(dataframe, ['Age', 'Price', 'Shipping delay in days', 'Arrival delay in days',
                                                'Product description accuracy', 'Manufacturer sustainability',
                                                'Packaging quality',
                                                'Additional options', 'Helpfulness of reviews and ratings',
                                                'Integrity of packaging',
                                                'Ease check-out procedure', 'Relevance of related products',
                                                'Costumer insurance'], False)

    # PCA
    pca2, dataframe_pca = pca(dataframe, verbose=True)  # At this point dataframe is only scaled

    test_model(dataframe)

if __name__ == '__main__':
    main()
