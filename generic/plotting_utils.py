"""
Plotting Utils, meaning functions for plotting for example the data after dimensionality reduction, clustered data,
missing values and more.
"""

import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
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


def plot_correlation_matrix(data, save_fig=False):
    """
    Plot correlation matrix between numeric features- in our case all features are numeric.
    :param data: Our data
    :return: Correlation matrix plot
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Correlation Matrix Between Numeric Features", y=1.03)
    sns.heatmap(data.corr(), annot=False, cmap='twilight', xticklabels=data.columns, yticklabels=data.columns)
    plt.xticks(rotation=90)
    if save_fig:
        fig.savefig("Figures/correlation_matrix.svg", format='svg')
        plt.close(fig)
        return
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

def get_method_description(method):
    if method.lower() == 'pca':
        method = 'PCA'
        data_prefix = 'PC'
        explanation_prefix = 'Prinicipal Component'
    elif method.lower() == "tsne":
        method = 't-SNE'
        data_prefix = 'Dim'    
        explanation_prefix = 'Dimension'
    else: 
        raise ValueError("Method is not supported")
    
    return method, data_prefix, explanation_prefix

def plot_dim_reduction(pca_df,target_column= None, labels_dict = None, save_fig=False, fig_name = None, method = 'PCA'):
    '''
    Plots pca analysis via scatter plot for 2d/3d data.
    Args:
        pca_df (pd.DataFrame): Data frame with PCA results
        target_column (str): Column name of the target column
        save_fig (bool): Save the figure
        fig_name (str): Name of the figure
        method (str): method type ('pca','tsne')
    Returns:
        None: plot the PCA results/ save the figure'''
    components = pca_df.columns
    n_components = len(components)
    if target_column:
        if target_column not in pca_df.columns:
            raise ValueError("Target column is not in the data frame")
        n_components -= 1
    method,data_prefix,explanation_prefix = get_method_description(method)
    
    if n_components == 2:
        plot_2d_scatter(pca_df,target_column, labels_dict, save_fig, fig_name, method, data_prefix, explanation_prefix)
    elif n_components == 3:
        plot_3d_scatter(pca_df,target_column, labels_dict, save_fig, fig_name, method, data_prefix, explanation_prefix)
    else:
        raise ValueError("Number of components is not supported for plotting")

def plot_2d_scatter(pca_df,target_column= None, label_dict=None, save_fig=False, fig_name = None,
                     method = 'PCA', prefix = 'PC', explanation_prefix = 'Prinicipal Component'):
    '''
    Plot PCA analysis with marked samples from the target column if given
    Args:
        pca_df (pd.DataFrame): Data frame with PCA results
        target_column (str): Column name of the target column
        save_fig (bool): Save the figure
        fig_name (str): Name of the figure
        method (str): method type ('pca','tsne') 
        prefix (str): ('PC','Dim') 
    Returns:
        None: plot the PCA results
    '''
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_df[f'{prefix}1'], pca_df[f'{prefix}2'],
                 c=pca_df[target_column] if target_column in pca_df.columns else None, cmap='viridis', alpha=0.7)
    plt.title(f'{method} Result')
    plt.xlabel(f'{explanation_prefix} 1')
    plt.ylabel(f'{explanation_prefix} 2')
    if label_dict:
        plt.colorbar(label=label_dict)
    else:
        plt.colorbar(label=target_column if target_column in pca_df.columns else 'None')
    if save_fig:
        output_path = f"Figures/{method}_results_2d.svg"
        if fig_name:
            output_path = f"Figures/{method}_results_2d_{fig_name}.svg"
        plt.savefig(output_path, format='svg')
        plt.close()
        return
    plt.show()

def plot_3d_scatter(pca_df, target_column = None, save_fig = False, fig_name=None, 
                    method = 'PCA', prefix = 'PC', explanation_prefix = 'Prinicipal Component'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot data
    scatter = ax.scatter(pca_df[f'{prefix}1'], pca_df[f'{prefix}2'],pca_df[f'{prefix}3'], 
                         c=pca_df[target_column] if target_column in pca_df.columns else None, cmap='viridis', alpha=0.7)
    # Labels and title
    ax.set_xlabel(f'{explanation_prefix} 1')
    ax.set_ylabel(f'{explanation_prefix} 2')
    ax.set_zlabel(f'{explanation_prefix} 3')
    ax.set_title(f'{method} Result')
    
    fig.colorbar(scatter, ax=ax, label=target_column if target_column in pca_df.columns else 'None')

    if save_fig:
        output_path = f"Figures/{method}_results_3d.svg"
        if fig_name:
            output_path = f"Figures/{method}_results_3d_{fig_name}.svg"
        plt.savefig(output_path, format='svg')
        plt.close()
        return
    plt.show()

def plot_2d_scatter_on_ax(pca_df, target_column=None, labels = None, save_fig=False, fig_name=None,
                     method='PCA', prefix='PC', explanation_prefix='Principal Component', ax=None, ax_title=None):
    '''
    Plot PCA analysis with marked samples from the target column if given
    Args:
        pca_df (pd.DataFrame): Data frame with PCA results
        target_column (str): Column name of the target column
        save_fig (bool): Save the figure
        fig_name (str): Name of the figure
        method (str): method type ('pca','tsne') 
        prefix (str): ('PC','Dim') 
        explanation_prefix (str): Explanation for axis labels
        ax (matplotlib.axes.Axes): Axes object for subplot
    Returns:
        None: plot the PCA results
    '''
    if ax is None:
        ax = plt.gca()  # If no ax provided, get current axis
    target_vals = pca_df[target_column] if target_column in pca_df.columns else None
    
    
    scatter = ax.scatter(pca_df[f'{prefix}1'], pca_df[f'{prefix}2'],
                         c=target_vals,
                         cmap='viridis', alpha=0.7)
    ax.set_title(f'{ax_title} Result')
    ax.set_xlabel(f'{explanation_prefix} 1')
    ax.set_ylabel(f'{explanation_prefix} 2')
    
    # Color bar if target_column is present
    if target_column in pca_df.columns:
        fig = plt.gcf()
        cbar = fig.colorbar(scatter, ax=ax)
        if labels:
            cbar.set_ticks(range(len(labels)))  # Set ticks for each unique value


            try:
                cbar.set_ticklabels([labels[val] for val in cbar.get_ticks()])
            except Exception as e:
                print(labels)
        else:
            cbar.set_label(target_column)

    # Saving the figure if needed
    if save_fig:
        output_path = f"Figures/{method}_results_2d.svg"
        if fig_name:
            output_path = f"Figures/{method}_results_2d_{fig_name}.svg"
        plt.savefig(output_path, format='svg')
        plt.close()
        return
    
    return ax

def sub_plot_dim(data_list, titles, generall_title,
                 method='PCA'):
    method,data_prefix,explanation_prefix = get_method_description(method)

    import math
    num_plots = len(data_list)
    num_cols = math.ceil(math.sqrt(num_plots))  # Columns = ceil(sqrt(number of dataframes))
    num_rows = math.ceil(num_plots / num_cols)  # Rows = ceil(number of dataframes / num_cols)

    # Create the subplots
    fig, axs = plt.subplots(num_rows, num_cols, sharex='all',sharey='all',figsize = (3*num_cols,3*num_rows) )

    # Flatten the axs array for easy iteration if it's a 2D array
    axs = axs.flatten()
    
    for index,(data_tit) in enumerate(zip(data_list,titles)):
        data_tup,title = data_tit
        data,target_column,labels = data_tup
        ax = axs[index]
        ax = plot_2d_scatter_on_ax(data, target_column=target_column,labels=labels, ax=ax, ax_title=title,
                                   method=method,prefix=data_prefix, explanation_prefix= explanation_prefix)
    for j in range(index + 1, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()  # Adjust layout to avoid overlapping subplots
    #plt.savefig(f'Figures/{generall_title}.svg', format='svg')
    plt.savefig(f'Figures/{generall_title}.png', dpi=300, format='png')
