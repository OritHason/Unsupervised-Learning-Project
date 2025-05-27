from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from scipy.stats import f_oneway, kruskal
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
import pandas as pd
import matplotlib.cm as cm

from generic.preprocessing import *

def hirearchial_clustering(data, method='ward', metric='euclidean',n_clusters=3,only_labels=True,random_state=None):
    Z = linkage(data, method=method)
    clusters = fcluster(Z, n_clusters, criterion='maxclust')
    if only_labels:
        return clusters
    return Z, clusters
def cluster(data, method,**kwargs):
    """
    Cluster data with the given method.
    Args:
        data: Data to cluster
        method: Clustering method to use
    """
    if method == 'kmeans':
        return kmeans(data, **kwargs)
    elif method == 'dbscan':
        return dbscan(data, **kwargs)
    elif method == 'gmm':
        return gmm(data, **kwargs)
    elif method == 'hierarchical':
        return hirearchial_clustering(data, **kwargs)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
def dbscan(data, eps = 0.5, min_samples = 5):
    """
    Density-based spatial clustering of applications with noise (DBSCAN).
    """
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    return db.labels_
def kmeans(data, n_clusters = 4):
    """
    KMeans clustering.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    return kmeans.labels_
def gmm(data, n_clusters = 4, random_state = 42):
    """
    Gaussian Mixture Model clustering.
    """
    gmm = GaussianMixture(n_components=n_clusters, random_state=random_state).fit(data)
    return gmm.predict(data)
def compare_intersect_of_cluster(cluster_labels, other_labels):
    """
    Compare the intersection of two clustering/PCA labels.
    The labels should be in the same order.
    i.e. each data point should have the same index in both labels.
    
    Args:
        cluster_labels: list of cluster labels
        other_labels: list of other labels
    """
    merged_data = pd.DataFrame({'cluster_labels': cluster_labels, 'other_labels': other_labels})
    cross_tab_by_cluster = pd.crosstab(merged_data['cluster_labels'], merged_data['other_labels'], normalize='index')
    cross_tab_by_label = pd.crosstab(merged_data['other_labels'], merged_data['cluster_labels'], normalize='index')
    max_keys_by_cluster = cross_tab_by_cluster.idxmax()
    max_values_by_cluster = cross_tab_by_cluster.max()
    max_keys_by_label = cross_tab_by_label.idxmax()
    max_values_by_label = cross_tab_by_label.max()
    summary_data_frame = pd.DataFrame({'max_keys_by_cluster': max_keys_by_cluster, 'max_values_by_cluster': max_values_by_cluster, 'max_keys_by_label': max_keys_by_label, 'max_values_by_label': max_values_by_label})
    return summary_data_frame


def analyze_feature_diffriniation_both_cato_no_cato(categorical_anova_path = None, no_categorical_anova_path = None, remove_remote_work = True, cluster_method='gmm'):
    cato_anova = pd.read_csv(categorical_anova_path,index_col='feature') if categorical_anova_path else None
    no_cato_anova = pd.read_csv(no_categorical_anova_path,index_col='feature') if no_categorical_anova_path else None
    data,features_in_data = get_working_data()
    if remove_remote_work:
        data = data.drop(columns=['Remote_Work'])
        features_in_data = {feature: feature_type for feature, feature_type in features_in_data.items() if feature != 'Remote_Work'}
    numeric_features = [feature for feature,feature_type in features_in_data.items() if feature_type == FeatureType.NUMBER]
    categorical_features = [feature for feature,feature_type in features_in_data.items() if feature_type == FeatureType.CATEGORY]
    # remove categorical
    data_no_categorical = remove_features_from_data(data, categorical_features)
    features_in_data_no_categorical = {feature: feature_type for feature, feature_type in features_in_data.items() if feature not in categorical_features}
    processed_data_no_categorical,t_,tc_ = preprocess_data_for_dim(data_no_categorical,target_column=None,data_features=features_in_data_no_categorical,
                                             remove_target=False)
    processed_data_with_cato,t_,tc_ = preprocess_data_for_dim(data,target_column=None,data_features=features_in_data,
                                             remove_target=False)
    numeric_features_no_cato = numeric_features 
    numeric_features_with_cato = numeric_features + [f'col_{i}' for i in range(4)]
    processed_data_with_cato.columns = [f'col_{i}' if i < 4 else processed_data_with_cato.columns[i] for i in range(len(processed_data_with_cato.columns))]
    processed_data_no_categorical_ = processed_data_no_categorical.copy()
    processed_data_with_cato_ = processed_data_with_cato.copy()
    processed_data_no_categorical = dim_reduction(processed_data_no_categorical, method='pca', n_components=2)
    processed_data_with_cato = dim_reduction(processed_data_with_cato, method='pca', n_components=2)
    cluster_labels_cato = cluster(processed_data_with_cato, cluster_method, n_clusters=3,random_state=(10 + 324%1))
    cluster_labels_no_cato = cluster(processed_data_no_categorical, cluster_method, n_clusters=3,random_state=(10 + 324%1))
    processed_data_no_categorical[numeric_features_no_cato] = processed_data_no_categorical_[numeric_features_no_cato]
    processed_data_with_cato[numeric_features_with_cato] = processed_data_with_cato_[numeric_features_with_cato]
    processed_data_no_categorical['cluster_labels'] = cluster_labels_no_cato
    processed_data_with_cato['cluster_labels'] = cluster_labels_cato
    
    return (processed_data_with_cato,processed_data_no_categorical,numeric_features_with_cato,
            numeric_features_no_cato,cato_anova,no_cato_anova)
    




def analyze_feature_diffrintiation_per_cluster(cluster_method, num_clusters, data_name = '', 
                                           reduce_dim = True, dim_method = 'pca', dim_components = 2,
                                           random_inits = 1,remove_categorical = True, remove_remote_work =True):
    data,features_in_data = get_working_data()
    # REMOVE REMOTE WORK
    if remove_remote_work:
        data = data.drop(columns=['Remote_Work'])
        features_in_data = {feature: feature_type for feature, feature_type in features_in_data.items() if feature != 'Remote_Work'}
    numeric_features = [feature for feature,feature_type in features_in_data.items() if feature_type == FeatureType.NUMBER]
    cato_suffix = 'with_categorical'
    categorical_features = [feature for feature,feature_type in features_in_data.items() if feature_type == FeatureType.CATEGORY]
    # remove categorical
    
    if remove_categorical:
        data = remove_features_from_data(data, categorical_features)
        features_in_data = {feature: feature_type for feature, feature_type in features_in_data.items() if feature not in categorical_features}
        cato_suffix = 'without_categorical'
    processed_data,t_,tc_ = preprocess_data_for_dim(data,target_column=None,data_features=features_in_data,
                                             remove_target=False)
    if not remove_categorical:
        numeric_features = numeric_features + [f'col_{i}' for i in range(4)]
        processed_data.columns = [f'col_{i}' if i < 4 else processed_data.columns[i] for i in range(len(processed_data.columns))]
    
    data_suffix = f'{cato_suffix}_processed'
    if reduce_dim:
        processed_data_ = processed_data.copy()
        processed_data = dim_reduction(processed_data, method=dim_method, n_components=dim_components)
        data_suffix = f'{cato_suffix}_reduced'
    data_name = data_name + '_' + data_suffix
    anovas_list = []
    for i in range(random_inits): # 10 different random inits with reproducibility
        cluster_labels = cluster(processed_data, cluster_method, n_clusters=num_clusters,random_state=((i+1)*10 + 324%(i+1)))
        if reduce_dim:
            processed_data[numeric_features] = processed_data_[numeric_features]
        processed_data['cluster_labels'] = cluster_labels
        
        unique_clusters = set(cluster_labels)
        anova_df = annova_testing(numeric_features, 'cluster_labels', processed_data, unique_clusters)   
        anovas_list.append(anova_df)
    merged_anova = merge_f_oneway_to_dataframe(anovas_list)
    merged_anova_summarized = summarize_merged_annova(merged_anova)
    merged_anova_summarized.to_csv(os.path.join(os.getcwd(),'generic', 'Statistics', f'{data_name}_{cluster_method}_anova_results.csv'))
   

def summarize_merged_annova(merged_anova):
    """
    Summarize the merged ANOVA results.
    Args:
        merged_anova: Merged ANOVA results
    """
    merged_anova['f_stats_mean'] = merged_anova['f_stats'].apply(lambda x: np.mean(x))
    merged_anova['f_stats_std'] = merged_anova['f_stats'].apply(lambda x: np.std(x))
    merged_anova['f_stats_median'] = merged_anova['f_stats'].apply(lambda x: np.median(x))
    merged_anova['p_vals_mean'] = merged_anova['p_vals'].apply(lambda x: np.mean(x))
    merged_anova['p_vals_std'] = merged_anova['p_vals'].apply(lambda x: np.std(x))
    merged_anova['p_vals_median'] = merged_anova['p_vals'].apply(lambda x: np.median(x))
    return merged_anova[['f_stats_mean', 'f_stats_std', 'f_stats_median', 'p_vals_mean', 'p_vals_std', 'p_vals_median']].sort_values(by='f_stats_mean', ascending=False)
def merge_f_oneway_to_dataframe(dfs):
    # Sort all dataframes by index
    dfs = [df.sort_index() for df in dfs]
    
    # Get feature names from the first dataframe
    features = dfs[0].index

    # Create a dictionary to build the DataFrame
    data = {
        'feature': [],
        'f_stats': [],
        'p_vals': []
    }

    for feature in features:
        f_stats = np.array([df.loc[feature, 'f_stat'] for df in dfs])
        p_vals = np.array([df.loc[feature, 'p_value'] for df in dfs])
        data['feature'].append(feature)
        data['f_stats'].append(f_stats)
        data['p_vals'].append(p_vals)

    return pd.DataFrame(data).set_index('feature')


def annova_testing(numeric_features, cluster_column, data_frame, num_clusters):
    """
    Apply ANOVA test to the numeric features in the data frame.
    
    Args:
        numeric_features (list): list of numeric features to test
        cluster_column (str): name of the column containing the cluster labels
        data_frame (pd.DataFrame): data frame containing the data
        num_clusters (list): unique clusters identifiers
    
    Returns:
        pd.DataFrame: data frame containing the F-statistic and p-value for each feature
    """
    annova_df = pd.DataFrame(index=numeric_features, columns=['f_stat', 'p_value'])
    for feature in numeric_features:
        groups = [data_frame[data_frame[cluster_column] == i][feature] for i in num_clusters]
        f_stat, p_value = f_oneway(*groups)  # ANOVA test
        annova_df.loc[feature] = [f_stat, p_value]
    return annova_df
def kruskal_testing(numeric_features, cluster_column, data_frame, num_clusters):
    """ """
    kruskal_df = pd.DataFrame(index=numeric_features, columns=['f_stat', 'p_value'])
    for feature in numeric_features:
        groups = [data_frame[data_frame[cluster_column] == i][feature] for i in range(num_clusters)]
        H_stat, p_value = kruskal(*groups)  # ANOVA test
        kruskal_df.loc[feature] = [H_stat, p_value]
    return kruskal_df

def plot_clustering_with_pca():
    
    X, X_ =main_working_data()
    target_data = X['Remote_Work']
    X = X.drop(columns=['Remote_Work'])
    clusters = 5
    n_clusters = 3
    for cluster in range(clusters):

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        # ax2.set_xlim([-0.1, 1])
        # # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # # plots of individual clusters, to demarcate them clearly.
        # ax2.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        #clusterer = GaussianMixture(n_components=n_clusters, random_state=(cluster+1)*10)
        clusterer = KMeans(n_clusters=n_clusters, random_state=(cluster+1)*10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )
        target_data,labels_col = convert_categorial_labels_into_numbers(target_data,'Remote_Work')

        scatter = ax1.scatter(X['PC1'], X['PC2'],
                 c=target_data, cmap='viridis', alpha=0.7)
        ax1.set_title(f'Remote labeled')
        ax1.set_xlabel(f'pc 1')
        ax1.set_ylabel(f'pc 2')
        if labels_col:
            cbar = ax1.figure.colorbar(scatter, ax=ax1)
            cbar.set_label(labels_col)
        
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            X['PC1'], X['PC2'], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            % n_clusters,
            fontsize=14,
            fontweight="bold",
        )

    plt.show()


def get_avg_siloute_score(X, n_clusters = None, method = None,**kwargs):
    """
    Get the average silhouette score for the given data and clustering method.
    Args:
        X: Data to cluster
        n_clusters: Number of clusters
        method: Clustering method to use
    """
    if method == 'hierarchical':
        kwargs['only_labels'] = True
    cluster_labels = cluster(X, method, n_clusters=n_clusters, **kwargs)
    avg_siloute = silhouette_score(X, cluster_labels)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    return (avg_siloute, sample_silhouette_values,cluster_labels)
def plot_clustering():
    """
    Plot clustering and sillouette for given clustering algorithem
    """
    data, feature_in_data =get_working_data()
    # remove categorical
    removed_features = ['Remote_Work','Work_Life_Balance','Job_Level']
    data = remove_features_from_data(data, removed_features)
    features_in_data = {feature: feature_type for feature, feature_type in feature_in_data.items() if feature not in removed_features}
    
    processed_data,t_,tc_ = preprocess_data_for_dim(data,target_column=None,data_features=features_in_data,
                                             remove_target=False)
    dim =  dim_reduction(processed_data, method='pca', n_components=2)
    X = dim
    range_n_clusters = [i for i in range(2,10)]
    for n_clusters in range_n_clusters:

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = GaussianMixture(n_components=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        # ax2.scatter(
        #     X['PC1'], X['PC2'], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        # )

        # # Labeling the clusters
        # centers = clusterer.means_
        # # Draw white circles at cluster centers
        # ax2.scatter(
        #     centers[:, 0],
        #     centers[:, 1],
        #     marker="o",
        #     c="white",
        #     alpha=1,
        #     s=200,
        #     edgecolor="k",
        # )

        # for i, c in enumerate(centers):
        #     ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        # ax2.set_title("The visualization of the clustered data.")
        # ax2.set_xlabel("Feature space for the 1st feature")
        # ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            % n_clusters,
            fontsize=14,
            fontweight="bold",
        )

    plt.show()



if __name__ == '__main__':
    #reduced_data, standarized_data =main_working_data()
    # l_1 = dbscan(standarized_data,eps = 0.1, min_samples=100)
    # l_2 = dbscan(reduced_data,eps = 0.1, min_samples=100)
    # print(l_1[:20])
    # print(l_2[:20])
    # l_3 = kmeans(standarized_data, n_clusters=3)
  
    # l_4 = kmeans(reduced_data, n_clusters=3)
    #plot_clustering()
    analyze_feature_diffrintiation_per_cluster(cluster_method='gmm',num_clusters=3,
                                           data_name='Working data')
    analyze_feature_diffriniation_both_cato_no_cato()
    

