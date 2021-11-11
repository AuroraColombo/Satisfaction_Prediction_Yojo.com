import math

from data.load_dataset import load_dataset
from data.preparation import *
import matplotlib.pyplot as plt
import seaborn as sns

df = load_dataset('model.csv', True)
df.drop(labels='id', axis=1, inplace=True)

df = remove_duplicates(df, True)

df, _ = remove_missing_values(df, True)

# df_categorical = df.select_dtypes(include=['object'])
# df = categorical_to_dummy(df, True)
# columns = df.columns

plt.show()
var = "Ease check-out procedure"
df[var].hist(bins=10)
plt.show()

# X0 = df[df['Satisfied'] == 0]
# X1 = df[df['Satisfied'] == 1]

# sns.histplot(X0[var], color="blue", stat='density', element="step", alpha=0.3)
# sns.histplot(X1[var], color="red", stat='density', element="step", alpha=0.3)
# plt.show()

sns.boxplot(x=df[var])
plt.show()
df = feature_2_log(df, var, 10)
sns.boxplot(x=df[var])
plt.show()
# # df, _ = standardize(df)
# sns.boxplot(x=df['Price'])
# plt.show()
'''
X0 = df[df['Satisfied'] == 0]
X1 = df[df['Satisfied'] == 1]

sns.histplot(X0[var], color="blue", stat='density', element="step", alpha=0.3)
sns.histplot(X1[var], color="red", stat='density', element="step", alpha=0.3)
plt.show()

# categoriche


df_0 = df_categorical[df['Satisfied'] == 0]  # records wih target==0
df_1 = df_categorical[df['Satisfied'] == 1]  # records wih target==1

fig, axes = plt.subplots(2, 3, figsize=[15, 6])
axes = axes.flatten()
fig.tight_layout(pad=2)

i = 0
for x in df_categorical.columns:
    plt.sca(axes[i])  # set the current Axes
    plt.hist([df_0[x], df_1[x]], density=True)
    plt.title(x)
    i += 1
plt.show()
'''
