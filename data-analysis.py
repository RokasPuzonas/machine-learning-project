import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
import seaborn as sns

dataset = pd.read_csv('data/counted-trimmed.csv')
dataset_target = dataset["target"]
dataset = dataset.drop(columns=["target"])

# Apply logarithmic scaling to all columns
for column_name in dataset.columns:
    dataset[column_name] = np.log(dataset[column_name] + 1)

# Drop all columns below average for visualization
average_count = 0
rare_columns = []
for column_name in dataset.columns:
    average_count += sum(dataset[column_name])
average_count /= len(dataset.columns)

rare_columns = []
for column_name in dataset.columns:
    if sum(dataset[column_name]) < average_count:
        rare_columns.append(column_name)
dataset = dataset.drop(columns=rare_columns)

# Sort columns from most used to least used
histogram_values = []
for column_name in dataset.columns:
    histogram_values.append(sum(dataset[column_name]))

histogram_columns = dataset.columns
histogram_tuples = list(zip(histogram_columns, histogram_values))
histogram_tuples.sort(key=lambda r: -r[1])
histogram_columns = list(r[0] for r in histogram_tuples)
histogram_values = list(r[1] for r in histogram_tuples)

# Plot frequency histogram
if False:
    # Draw histogram
    plt.bar(histogram_columns, histogram_values)
    plt.xticks(histogram_columns, rotation='vertical')
    plt.show()

# Plot density plot of the most common functions (top 5)
if False:
    targets = dataset_target.unique()
    cmap = colormaps.get_cmap('hsv').resampled(len(targets)+1)

    columns = histogram_columns[:5]
    for column_name in columns:
        for i, target_name in enumerate(targets):
            dataset[dataset_target == target_name][column_name].plot.density(label=target_name, c=cmap(i))

        plt.legend()
        plt.title(f"'{column_name}' density plot")
        plt.xlim(-5, 15)
        plt.show()

# Correlation matrix
if True:
    sns.heatmap(dataset.corr())
    plt.title("Correlation matrix")
    plt.show()