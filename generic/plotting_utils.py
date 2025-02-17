"""
Plotting Utils, meaning functions for plotting for example the data after dimensionality reduction, clustered data,
missing values and more.
"""

import numpy as np
import pandas as pd
# from mpl_toolkits.mplot3d import Axes3D
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import matplotlib as mpl


def plot_missing_values(data, save_fig=False):
    """
    Some features has missing values. This function plots all of them and displays the percentage of missing values.
    :param data: (pd.DataFrame) Our data set.
    :param save_fig: (bool):Specify whether the output figure will be saved.
    :return: plot
    """
    # Get percentage of missing values
    missing_values_columns = data.isnull().mean().sort_values(ascending=False) * 100
    missing_values = pd.DataFrame({'available': 100 - missing_values_columns, 'missing': missing_values_columns})

    # --- Plotting --- #

    figure, ax = plt.subplots(figsize=(9, 5))

    y_height = list(reversed(range(len(missing_values))))
    ax.barh(y_height, missing_values['available'], color='C0')
    ax.barh(y_height, missing_values['missing'], left=missing_values['available'], color='C3')
    ax.set_yticks(y_height)
    ax.set_yticklabels(missing_values.index.values)
    ax.set_ylabel("feature")
    ax.set_xlabel("percentage")
    ax.set_title("Features with missing values")
    ax.legend(['available', 'missing'])

    # Add percent text
    for p in ax.patches:
        if p.get_x() > 0:
            width = p.get_width()

            # Arguments: xPos, yPos, text, alignment
            ax.text(50,
                    p.get_y() + p.get_height() / 2,
                    '{:.2f}%'.format(width) + " missing",
                    ha="center", va="center")

    figure.tight_layout()
    plt.show()

    if save_fig:
        figure.savefig("Figures/missing_values.eps", format='eps')


def plot_correlation_matrix(data):
    """
    Plot correlation matrix between numeric features- in our case all features are numeric.
    :param data: Our data
    :return: Correlation matrix plot
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Correlation Matrix Between Numeric Features", y=1.03)
    sns.heatmap(data.corr(), annot=True, cmap='twilight', xticklabels=data.columns, yticklabels=data.columns)
    plt.xticks(rotation=40)
    plt.show()


def plot_numeric_features_distribution(data, num_dtypes_columns):
    """
    Plot the distribution of numeric features. For all numeric features, for each value, we plot the density
    of that value.
    :param data: (DataFrame) Our data set
    :param num_dtypes_columns: (list) List of the names of the numeric features.
    :return: Plots of distributions
    """

    def get_bins(series):
        """Get the number of desired bins for this series."""

        bins = max(series) - min(series)
        bins = int(bins)

        if bins > 40:
            bins = 40
        if bins < 10:
            bins = 10

        return bins

    fig, ax = plt.subplots(5, 4, figsize=(16, 10))

    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            k = i * ax.shape[1] + j  # track the current feature to plot

            # Make sure we do not exceed the number of features we have
            if k < len(num_dtypes_columns):
                curr_col_name = num_dtypes_columns[k]
                curr_col = data[curr_col_name]
                print(curr_col_name, "| bins =", get_bins(curr_col))  # print bin size

                sns.distplot(curr_col, bins=get_bins(curr_col), norm_hist=True, kde=False, ax=ax[i, j])
            else:
                ax[i, j].axis('off')

        ax[i, 0].set_ylabel("Density")

    fig.tight_layout()
    plt.show()


def do_boxplot(data):
    """
    Plot a box plot for outliers visualization
    :param data: Our data set
    :return: A box plot
    """
    X = data.values
    names = data.columns
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)  # normalize
    scaled_df = pd.DataFrame(X_scaled, columns=names)
    plt.figure(figsize=(10, 10))
    plt.title("Box Plot")
    sns.boxplot(data=scaled_df)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.show()
