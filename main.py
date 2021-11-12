import pickle

from data.load_dataset import load_dataset
from data.preparation import *
from model.HeterogeneousEnsemble import HeterogeneousEnsemble


def main():
    # %% Loading the dataset
    dataframe = load_dataset('model.csv', False)

    # %% Preparing the dataset
    # We drop the attribute because of the high correlation with Arrival delay in days
    dataframe.drop(labels=['id', 'Shipping delay in days'], axis=1, inplace=True)

    # Removing eventual identical records
    dataframe = remove_duplicates(dataframe, False)

    # Removing records with at least one missing values
    _, df_wo_nan = remove_missing_values(dataframe, False)

    # Converting the selected features to dummy variables
    df_wo_nan = categorical_to_dummy(df_wo_nan, variables2convert=['Gender', 'Customer Type', 'NewUsed', 'Category',
                                                                   'Satisfaction'], verbose=False)

    # Normalizing variable 'Price' through log10 transformation
    df_wo_nan = feature_2_log(df_wo_nan, 'Price', 10)

    # StandardScaler only to numerical variables
    df_wo_nan, scaler = standardise(df_wo_nan, ['Age', 'Price', 'Arrival delay in days',
                                                'Product description accuracy', 'Manufacturer sustainability',
                                                'Packaging quality',
                                                'Additional options', 'Helpfulness of reviews and ratings',
                                                'Integrity of packaging',
                                                'Ease check-out procedure', 'Relevance of related products',
                                                'Costumer insurance'], False)

    pickle.dump(scaler, open('scaler.pkl', 'wb'))

    # %% Heterogeneous Ensemble method,

    # Instance of the model and testing via cross validation
    het_ens = HeterogeneousEnsemble(threshold=2)
    het_ens.cross_validation(df_wo_nan)


if __name__ == '__main__':
    main()
