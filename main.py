from data.load_dataset import load_dataset
from data.preparation import *
from training.model_training import test_model


def main():
    dataframe = load_dataset('model.csv', False)
    dataframe.drop(labels='id', axis=1, inplace=True)
    dataframe = remove_duplicates(dataframe, False)
    dataframe, df_wo_nan = remove_missing_values(dataframe, False)
    dataframe = categorical_to_dummy(dataframe, False)

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

    # Paired plot
    # paired_plot(dataframe)
    # plt.show()

    # PCA
    pca2, dataframe_pca = pca(dataframe, verbose=False)  # At this point dataframe is only scaled

    # Models
    test_model(dataframe)


if __name__ == '__main__':
    main()
