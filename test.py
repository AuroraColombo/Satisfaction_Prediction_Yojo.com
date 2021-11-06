from data.load_dataset import load_dataset
from data.preparation import remove_duplicates, remove_missing_values, categorical_to_dummy

df = load_dataset('model.csv', True)
df = remove_duplicates(df, True)
_, df = remove_missing_values(df, True)
df = categorical_to_dummy(df, True)
