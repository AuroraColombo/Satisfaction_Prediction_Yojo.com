from data.load_dataset import load_dataset
from data.preparation import *
from training.model_training import test_model
from training.models import test_knn


def main():
    dataframe = load_dataset('model.csv', False)
    dataframe.drop(labels=['id', 'Shipping delay in days'], axis=1, inplace=True)
    dataframe = remove_duplicates(dataframe, False)
    dataframe, df_wo_nan = remove_missing_values(dataframe, False)

    dataframe01 = dataframe.copy(deep=True)
    dataframe01.drop(labels=['Manufacturer sustainability', 'NewUsed', 'Category'], axis=1, inplace=True)
    dataframe19 = dataframe01.copy(deep=True)
    dataframe19.drop(labels=['Age', 'Product description accuracy', 'Arrival delay in days', 'Gender'], axis=1, inplace=True)



    dataframe = categorical_to_dummy(dataframe, variables2convert=['Gender', 'Customer Type', 'NewUsed', 'Category',
                                                                   'Satisfaction'], verbose=False)
    df_wo_nan = categorical_to_dummy(df_wo_nan, variables2convert=['Gender', 'Customer Type', 'NewUsed', 'Category',
                                                                   'Satisfaction'], verbose=False)

    dataframe01 = categorical_to_dummy(dataframe01, variables2convert=['Gender', 'Customer Type', 'Satisfaction'],
                                       verbose=False)

    dataframe19 = categorical_to_dummy(dataframe19, variables2convert=['Customer Type', 'Satisfaction'], verbose=False)

    # Normalizing variable 'Price' through log10 transformation

    dataframe = feature_2_log(dataframe, 'Price', 10)
    df_wo_nan = feature_2_log(df_wo_nan, 'Price', 10)
    dataframe01 = feature_2_log(dataframe01, 'Price', 10)
    dataframe19 = feature_2_log(dataframe19, 'Price', 10)

    # StandardScaler only to numerical variables
    dataframe, scaler = standardize(dataframe, ['Age', 'Price', 'Arrival delay in days',
                                                'Product description accuracy', 'Manufacturer sustainability',
                                                'Packaging quality',
                                                'Additional options', 'Helpfulness of reviews and ratings',
                                                'Integrity of packaging',
                                                'Ease check-out procedure', 'Relevance of related products',
                                                'Costumer insurance'], False)

    df_wo_nan, scaler = standardize(df_wo_nan, ['Age', 'Price', 'Arrival delay in days',
                                                'Product description accuracy', 'Manufacturer sustainability',
                                                'Packaging quality',
                                                'Additional options', 'Helpfulness of reviews and ratings',
                                                'Integrity of packaging',
                                                'Ease check-out procedure', 'Relevance of related products',
                                                'Costumer insurance'], False)

    dataframe01, scaler = standardize(dataframe01, ['Age', 'Price', 'Arrival delay in days',
                                                    'Product description accuracy',
                                                    'Packaging quality',
                                                    'Additional options', 'Helpfulness of reviews and ratings',
                                                    'Integrity of packaging',
                                                    'Ease check-out procedure', 'Relevance of related products',
                                                    'Costumer insurance'], False)

    dataframe19, scaler = standardize(dataframe19, ['Price',
                                                    'Packaging quality',
                                                    'Additional options', 'Helpfulness of reviews and ratings',
                                                    'Integrity of packaging',
                                                    'Ease check-out procedure', 'Relevance of related products',
                                                    'Costumer insurance'], False)

    # Paired plot
    # paired_plot(dataframe, 'Satisfied')
    # plt.show()

    # PCA
    pca2, dataframe_pca = pca(dataframe, verbose=True)  # At this point dataframe is only scaled
    pca2, dataframe_pca01 = pca(dataframe01, verbose=True)  # At this point dataframe is only scaled
    pca2, dataframe_pca19 = pca(dataframe19, verbose=True)  # At this point dataframe is only scaled

    # Outliers
    # count_outliers_zindex(dataframe)
    # count_outliers_boxplots(dataframe)

    # Models
    test_model(dataframe01)

    # Testing knn
    # test_knn(dataframe, 60, 63, 1) # 61 is the optimal


if __name__ == '__main__':
    main()
