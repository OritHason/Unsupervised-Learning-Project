"""
Plotting Utils, meaning functions for plotting for example the data after dimensionality reduction, clustered data,
missing values and more.
"""
import math
import matplotlib.patches as mpatches
import numpy as np
from scipy.cluster.hierarchy import dendrogram
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import os

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


def plot_correlation_matrix(data, save_fig=False, ax=None, ax_title=None,show_ticks=True):
    """
    Plot correlation matrix between numeric features- in our case all features are numeric.
    :param data: Our data
    :return: Correlation matrix plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax_title = "Correlation Matrix Between Numeric Features"

    ax.set_title(ax_title, y=1.03)
    corr = data.corr()
    mask = np.zeros_like(corr, dtype=bool)
    np.fill_diagonal(mask, True)  # Mask the diagonal
    short_columns = [col.split('_')[0] for col in data.columns]

    sns.heatmap(corr, mask=mask, ax=ax, annot=False, cmap='coolwarm', 
                xticklabels=short_columns if show_ticks else False, 
yticklabels=short_columns if show_ticks else False, vmin=-1, vmax=1)

    if show_ticks:
        ax.tick_params(axis='x', labelrotation=90)
    else:
        ax.tick_params(left=False, bottom=False)  # hide tick marks

    if save_fig:
        fig.savefig("Figures/correlation_matrix.svg", format='svg')
        plt.close(fig)
        return

def plot_clustered_feature_distributions(data, cluster_column, numeric_features, save_fig=False
                             ,num_clusters = 0,method_name = '', data_name = '', max_cols=4):
    """
    Plot histograms of numeric features, colored by cluster.

    Args:
        data (pd.DataFrame): The dataframe with cluster and numeric features
        cluster_column (str): The name of the column that holds cluster labels
        numeric_features (list): List of numeric column names to plot
        if_save_fig (bool): Whether to save the figure
        data_name (str): Name of dataset for saving figure
        max_cols (int): Maximum columns in subplot grid
    """
    num_features = len(numeric_features)
    n_cols = max_cols
    n_rows = math.ceil(num_features / n_cols)

    

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    cluster_labels = sorted(data[cluster_column].unique())

    for i in range(n_rows):
        for j in range(n_cols):
            k = i * n_cols + j
            if k < num_features:
                feature = numeric_features[k]
                bins = get_bins(data[feature])

                for cluster in cluster_labels:
                    cluster_data = data[data[cluster_column] == cluster][feature]
                    sns.histplot(cluster_data, bins=bins, kde=False, stat='density',
                                 label=f'Cluster {cluster}', ax=ax[i, j], element='step', fill=False)

                ax[i, j].set_title(f'{feature} by Cluster')
                ax[i, j].set_xlabel(feature)
                ax[i, j].set_ylabel("Density")
                ax[i, j].legend()
            else:
                ax[i, j].axis('off')

    fig.tight_layout()

    if save_fig:
        fig_name = f'histograms_clusters_{method_name or "clustering"}_{num_clusters}_{data_name}.png'

        fig.savefig(os.path.join(f'Figures',fig_name), format='png', bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def get_bins(series):
        """Get the number of desired bins for this series."""

        bins = max(series) - min(series)
        bins = int(bins)

        if bins > 40:
            bins = 40
        if bins < 10:
            bins = 10

        return bins
def plot_numeric_features_distribution(data, num_dtypes_columns, if_save_fig=False,data_name = None):
    """
    Plot the distribution of numeric features. For all numeric features, for each value, we plot the density
    of that value.
    :param data: (DataFrame) Our data set
    :param num_dtypes_columns: (list) List of the names of the numeric features.
    :return: Plots of distributions
    """

    

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
    if if_save_fig:
        path = '/home/alon/Unsupervised learning/Unsupervised-Learning-Project/Figures'
        fig.savefig(f"{path}/{data_name} numeric_features_distribution.png", format='png')
        plt.close(fig)
        return


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

def box_plot_clustering_data(data, cluster_column, numeric_features, save_fig=False
                             ,num_clusters = 0,method_name = '', data_name = ''):
    """
    Plot box plot for each numeric feature for each cluster.
    Create a subplot of box plots.
    
    Args:
        data (pd.DataFrame): Data frame with clustering results
        cluster_column (str): Column name of the cluster column
        numeric_features (list): List of numeric features to plot
        save_fig (bool): Save the figure
        num_clusters (int): Number of clusters
        method_name (str): Name of the clustering method used
        """
    num_features = len(numeric_features)
    cols = 3  # number of subplots per row
    rows = math.ceil(num_features / cols)

    plt.figure(figsize=(5 * cols, 4 * rows))
    
    for i, feature in enumerate(numeric_features):
        plt.subplot(rows, cols, i + 1)
        sns.boxplot(x=cluster_column, y=feature, data=data)
        plt.title(feature)
        plt.xlabel('Cluster')
        plt.ylabel(feature)

    plt.tight_layout()
    
    if method_name:
        plt.suptitle(f'Boxplots of Features by Cluster - {method_name}', fontsize=16, y=1.02)
    
    if save_fig:
        fig_name = f'boxplots_clusters_{method_name or "clustering"}_{num_clusters}_{data_name}.svg'
        
        plt.savefig(os.path.join('Figures', fig_name), bbox_inches='tight', format='svg')
        plt.close()
    else:
        plt.show()
def plot_box_cluster_together(data_with_categorical, data_without_categorical, 
                              all_features, subset_features,
                              with_categorical_anova, without_categorical_anova,
                              feature_order=None, axes=None, fig=None):
    
    extra_features = list(set(all_features) - set(subset_features))
    df1_name = 'With categorical'
    df2_name = 'Without categorical'

    # Step 2: Melt subset features from both df1 and df2
    df1_subset_melt = data_with_categorical[['cluster_labels'] + subset_features].melt(
        id_vars='cluster_labels', var_name='feature', value_name='value')
    df1_subset_melt['source'] = df1_name

    df2_subset_melt = data_without_categorical[['cluster_labels'] + subset_features].melt(
        id_vars='cluster_labels', var_name='feature', value_name='value')
    df2_subset_melt['source'] = df2_name

    # Step 3: Melt extra features only from df1
    df1_extra_melt = data_with_categorical[['cluster_labels'] + extra_features].melt(
        id_vars='cluster_labels', var_name='feature', value_name='value')
    df1_extra_melt['source'] = df1_name

    # Step 4: Combine everything into one DataFrame
    combined_all = pd.concat([df1_subset_melt, df2_subset_melt, df1_extra_melt], ignore_index=True)

    # Use feature_order if provided, else default to subset_features + extra_features
    all_feature_list = feature_order if feature_order is not None else (subset_features + extra_features)
    cols=3
    num_features = len(all_feature_list)
    rows = math.ceil(num_features / cols)

    # If axes not provided, create them
    if axes is None or fig is None:
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows), squeeze=False)
    else: axes = np.array(axes[:9]).reshape(3, 3)

    for i, feature in enumerate(all_feature_list):
        row, col = divmod(i, cols)
        ax = axes[row][col]

        sns.boxplot(
            data=combined_all[combined_all['feature'] == feature],
            x='cluster_labels',
            y='value',
            hue='source' if feature in subset_features else None,
            ax=ax
        )

        # Stats and legend patches
        stats_categorical = with_categorical_anova.loc[feature]
        f1 = stats_categorical['f_stats_mean']
        p1 = stats_categorical['p_vals_mean']

        handles = [mpatches.Patch(color=sns.color_palette()[0], label=f'F: {f1:.2g}, P: {p1:.2g}')]

        if feature in subset_features:
            stats_non_categorical = without_categorical_anova.loc[feature] 
            f2 = stats_non_categorical['f_stats_mean']
            p2 = stats_non_categorical['p_vals_mean']
            handles.append(mpatches.Patch(color=sns.color_palette()[1], label=f'F: {f2:.2g}, P: {p2:.2g}'))

        ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.1), fontsize=10)
        ax.set_title(f"{feature}")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Value")

       
        
    # Global legend for data source
    df1_patch = mpatches.Patch(color=sns.color_palette()[0], label=df1_name)
    df2_patch = mpatches.Patch(color=sns.color_palette()[1], label=df2_name)

    fig.legend(handles=[df1_patch, df2_patch], loc='lower center', ncol=2,
               title='Data type', prop={'size': 14}, title_fontsize='16')

    # Remove unused axes
    for j in range(num_features, rows * cols):
        fig.delaxes(axes[j // cols][j % cols])
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
def plot_dendogram(Z, ax=None, ax_title=None):
    """
    Plot a dendrogram for hierarchical clustering.
    Args:
        Z (ndarray): The linkage matrix from hierarchical clustering.
        ax (matplotlib.axes.Axes): Axes object for subplot.
        ax_title (str): Title for the axes.
    Returns:
        None: plot the dendrogram
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    
    dendrogram(Z, ax=ax)
    ax.set_title(ax_title)
    plt.savefig(f'{ax_title}.png',format='png')  
    
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

def visualise_feature_coeffiecnts_pca(loadings=None, data=None,pca=None, ax =None):
    if pca is not None:
        loadings = pd.DataFrame(pca.components_, columns=data.columns, index=["PC1", "PC2"])
    
    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the arrows and feature names
    for feature in data.columns:
        ax.arrow(0, 0,
                 loadings.loc["PC1", feature],
                 loadings.loc["PC2", feature],
                 color='r', alpha=0.5,
                 head_width=0.02, length_includes_head=True)
        
        ax.text(loadings.loc["PC1", feature] * 1.15,
                loadings.loc["PC2", feature] * 1.15,
                feature, color='g', ha='center', va='center')

    # Set dynamic axis limits with padding
    x_max = max(abs(loadings.loc["PC1", :])) * 1.3
    y_max = max(abs(loadings.loc["PC2", :])) * 1.3

    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(-y_max, y_max)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA feature contributions")
    ax.grid(True)
    ax.axhline(0, color='grey', lw=1)
    ax.axvline(0, color='grey', lw=1)


    


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
    ax.set_title(ax_title)
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

def sub_plot_dim_categoric(data_list, titles, generall_title, method='PCA'):
    method,data_prefix,explanation_prefix = get_method_description(method)

    import math
    num_plots = len(data_list)
    num_cols = math.ceil(math.sqrt(num_plots))  # Columns = ceil(sqrt(number of dataframes))
    num_rows = math.ceil(num_plots / num_cols)  # Rows = ceil(number of dataframes / num_cols)

    # Create the subplots
    fig, axs = plt.subplots(num_rows, num_cols, sharex='all',sharey='all',figsize = (3*num_cols,3*num_rows) )

    # Flatten the axs array for easy iteration if it's a 2D array
    axs = axs.flatten()
    for index,(data_tup) in enumerate(zip(data_list,titles)):
        data,title = data_tup
        ax = axs[index]
        ax = plot_2d_scatter_on_ax(data,ax =ax,ax_title=title,method=method,
                                   prefix=data_prefix, explanation_prefix= explanation_prefix)
    for j in range(index + 1, len(axs)):
        axs[j].axis('off')
    plt.tight_layout()  # Adjust layout to avoid overlapping subplots
    path = '/home/alon/Unsupervised learning/Unsupervised-Learning-Project/Figures'
    plt.savefig(f'{path}/{generall_title}.png', dpi=300, format='png')

def plot_cluster_crosstab(df, ax=None, title="Cluster Distribution by Category", colors=None):
    """
    Plot a crosstab DataFrame with clusters on the x-axis and grouped bars per category,
    including zero-value bars for visibility.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    index = df.index.astype(str)
    categories = df.columns
    x = np.arange(len(index))  # cluster positions
    n_categories = len(categories)
    width = 0.8 / n_categories  # bar width

    if colors is None:
        colors = plt.cm.Set2.colors

    for i, cat in enumerate(categories):
        values = df[cat].fillna(0)
        bar_positions = x + i * width
        bars = ax.bar(bar_positions, values, width=width, label=cat,
                      color=colors[i % len(colors)], edgecolor='black')

        for bar in bars:
            bar_hieght = bar.get_height()
            if bar_hieght> 0:
                if bar_hieght < 0.9:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
                else:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.05,
                        f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
    # Set x-axis ticks
    ax.set_xticks(x + width * (n_categories - 1) / 2)
    ax.set_xticklabels(index)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Proportion")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend(title="Category",fontsize=6)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Add vertical grid lines *between* clusters
    for i in range(1, len(x)):
        xpos = (x[i] + x[i - 1] + width * (n_categories - 1)) / 2
        ax.axvline(x=xpos, linestyle='--', color='gray', alpha=0.3)
    

def plot_dual_silhouettes_from_values(ax, sil_values_pca, sil_values_nopca, labels_pca, labels_nopca):
    pca_color = 'skyblue'  # Color for PCA clusters
    nopca_color = 'lightcoral'  # Color for No PCA clusters

    y_lower = 10
    ax.set_xlim([-1, 1])
    ax.set_ylim([0, 2 * len(sil_values_pca) + 20])  # Assuming same len for both
    n_clusters = set(labels_pca).union(set(labels_nopca))  # Unique cluster labels
    # Loop over PCA and No PCA values
    for idx, (sil_values, labels, title, offset, color) in enumerate([
        (sil_values_pca, labels_pca, "PCA", 0, pca_color),
        (sil_values_nopca, labels_nopca, "No PCA", len(sil_values_pca) + 10, nopca_color)
    ]):
        y_lower = offset  # Reset y_lower for each plot
        for i in n_clusters:
            cluster_silhouette_vals = sil_values[labels == i]
            cluster_silhouette_vals.sort()
            size_cluster_i = cluster_silhouette_vals.shape[0]

            y_upper = y_lower + size_cluster_i

            # Fill the area with a single color for all clusters in PCA or No PCA
            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                cluster_silhouette_vals,
                facecolor=color,
                edgecolor=color,
                alpha=0.6,
                label=f"{title}" if i == 0 else None  # Add label only once in the legend
            )

            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, f"{i+1}", fontsize=8)
            y_lower = y_upper + 10  # Spacing between clusters

  

    # Final plot settings
    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Samples")
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([])
    ax.set_title("Silhouette plot")

    # Add legend
    ax.legend(loc='lower left', fontsize=8)
def sub_plot_dim(data_list, titles, generall_title,
                 method='PCA'):
    method,data_prefix,explanation_prefix = get_method_description(method)

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
    path = '/home/alon/Unsupervised learning/Unsupervised-Learning-Project/Figures'
    plt.savefig(f'{path}/{generall_title}.png', dpi=300, format='png')
