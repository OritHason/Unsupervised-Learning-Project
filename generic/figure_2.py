import string
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from generic.clustering import *
from generic.preprocessing import *
# from clustering import *
# from preprocessing import *

fig = plt.figure(figsize=(9,12))
gs = gridspec.GridSpec(3, 3, figure=fig)  # 3 rows, 3 columns

# 9 box plots 3x3 pictures.
axes = [] 
# Hirearchical clustering
axes.append(fig.add_subplot(gs[0, 0]))  # A
axes.append(fig.add_subplot(gs[0, 1]))    # B
axes.append(fig.add_subplot(gs[0, 2]))    # C
# Second row (3 subplots)
# GMM clustering
axes.append(fig.add_subplot(gs[1, 0]))    # D
axes.append(fig.add_subplot(gs[1, 1]))    # E
axes.append(fig.add_subplot(gs[1, 2]))    # F

# Silhouette plots
axes.append(fig.add_subplot(gs[2, 0]))  # G
axes.append(fig.add_subplot(gs[2, 1]))  # H
axes.append(fig.add_subplot(gs[2, 2]))  # I

# cluster_method = 'gmm'
cluster_method = 'hierarchical'

def plot_figure_2():
    data,features_in_data = get_working_data()

    remote_job_work = ['Remote_Work','Job_Level','Work_Life_Balance']
    data_no_remotejobwork = data.drop(columns=remote_job_work)
    feature_in_data_no_remotejobwork = features_in_data.copy()
    feature_in_data_no_remotejobwork = {k: v for k, v in feature_in_data_no_remotejobwork.items() if k not in remote_job_work}

    processed_data_no_remotejobwork,t,tc_ = preprocess_data_for_dim(data=data_no_remotejobwork,target_column=None,data_features=feature_in_data_no_remotejobwork)

    cluster_labels_hierarchical = cluster(data=processed_data_no_remotejobwork, n_clusters=3,method='hierarchical')
    cluster_labels_gmm = cluster(data=processed_data_no_remotejobwork, n_clusters=3,method='gmm')
    for ax_index,feature in enumerate(remote_job_work):
        labels = data[feature]
        merged_data = pd.DataFrame({'cluster_labels': cluster_labels_hierarchical, 'other_labels': labels})
        cross_tab_by_cluster = pd.crosstab(merged_data['cluster_labels'], merged_data['other_labels'], normalize='index')
        plot_cluster_crosstab(df=cross_tab_by_cluster,ax=axes[ax_index],title=feature)
        merged_data = pd.DataFrame({'cluster_labels': cluster_labels_gmm, 'other_labels': labels})
        cross_tab_by_cluster = pd.crosstab(merged_data['cluster_labels'], merged_data['other_labels'], normalize='index')
        plot_cluster_crosstab(df=cross_tab_by_cluster,ax=axes[ax_index+3],title=feature)
    


    processed_data,t,tc_ = preprocess_data_for_dim(data=data,target_column=None,data_features=features_in_data)
    reduced_data_pca = dim_reduction(processed_data, n_components=2, method='pca')

    #evaluate silloute
    categorical_numeric = split_data_into_categorical_and_numeric(data,features_in_data)
    numeric_df, numeric_features = categorical_numeric[1]
    numeric_features = {feature: FeatureType.NUMBER for feature in numeric_features}
    numeric_df,t,tc_ = preprocess_data_for_dim(data=numeric_df, target_column=None, data_features=numeric_features)
    numeric_pca = dim_reduction(numeric_df, n_components=2, method='pca')
    clusters = [i for i in range (2,10)]
    #siloute_all_pca,siloute_numeric_pca,siloute_all_no_pca,siloute_numeric_no_pca = [],[],[],[]
    siloute_all_pca_gmm,siloute_numeric_pca_gmm,siloute_all_no_pca_gmm,siloute_numeric_no_pca_gmm = [],[],[],[]
    siloute_all_pca_hierarchical,siloute_numeric_pca_hierarchical,siloute_all_no_pca_hierarchical,siloute_numeric_no_pca_hierarchical = [],[],[],[]
    for i in clusters:
        siloute_all_pca_gmm.append(get_avg_siloute_score(X=reduced_data_pca, n_clusters=i,method='gmm'))
        siloute_all_pca_hierarchical.append(get_avg_siloute_score(X=reduced_data_pca, n_clusters=i,method='hierarchical'))
        siloute_numeric_pca_gmm.append(get_avg_siloute_score(X=numeric_pca, n_clusters=i,method='gmm')[0])
        siloute_numeric_pca_hierarchical.append(get_avg_siloute_score(X=numeric_pca, n_clusters=i,method='hierarchical')[0])
        siloute_all_no_pca_gmm.append(get_avg_siloute_score(X=processed_data, n_clusters=i,method='gmm'))
        siloute_all_no_pca_hierarchical.append(get_avg_siloute_score(X=processed_data, n_clusters=i,method='hierarchical'))
        siloute_numeric_no_pca_gmm.append(get_avg_siloute_score(X=numeric_df, n_clusters=i,method='gmm')[0])
        siloute_numeric_no_pca_hierarchical.append(get_avg_siloute_score(X=numeric_df, n_clusters=i,method='hierarchical')[0])
    silloute_avgs_all_pca_gmm = [item[0] for item in siloute_all_pca_gmm]
    silloute_avgs_all_no_pca_gmm = [item[0] for item in siloute_all_no_pca_gmm]
    silloute_avgs_all_pca_hierarchical = [item[0] for item in siloute_all_pca_hierarchical]
    silloute_avgs_all_no_pca_hierarchical = [item[0] for item in siloute_all_no_pca_hierarchical]
        # siloute_all_pca.append(get_avg_siloute_score(X=reduced_data_pca, n_clusters=i,method=cluster_method))
        # siloute_numeric_pca.append(get_avg_siloute_score(X=numeric_pca, n_clusters=i,method=cluster_method)[0])
        # siloute_all_no_pca.append(get_avg_siloute_score(X=processed_data, n_clusters=i,method=cluster_method))
        # siloute_numeric_no_pca.append(get_avg_siloute_score(X=numeric_df, n_clusters=i,method=cluster_method)[0])
    # silloute_avgs_all_pca = [item[0] for item in siloute_all_pca]
    # siloute_avgs_all_no_pca = [item[0] for item in siloute_all_no_pca]
    a1,a2 = axes[6],axes[7]
    # PCA subplot
    custom_lines = [
    Line2D([0], [0], color='gray', marker='o',  label='GMM'),
    Line2D([0], [0], color='gray', marker='s',  label='Hierarchical'),
    Line2D([0], [0], color=sns.color_palette()[0], linestyle='--', label='With categorical'),
    Line2D([0], [0], color=sns.color_palette()[1],  linestyle='-', label='Without categorical'),
    ]

    a1.plot(clusters, silloute_avgs_all_pca_gmm,  color=sns.color_palette()[0], marker='o',linestyle='--')
    a1.plot(clusters, siloute_numeric_pca_gmm, color=sns.color_palette()[1], marker='o',linestyle='-')
    a1.plot(clusters, silloute_avgs_all_pca_hierarchical, color=sns.color_palette()[0], marker='s',linestyle='--')
    a1.plot(clusters, siloute_numeric_pca_hierarchical,color=sns.color_palette()[1], marker='s',linestyle='-')
    
    # No PCA subplot
    a2.plot(clusters, silloute_avgs_all_no_pca_gmm, color=sns.color_palette()[0], marker='o',linestyle='--')
    a2.plot(clusters, siloute_numeric_no_pca_gmm,  color=sns.color_palette()[1], marker='o',linestyle='-')
    a2.plot(clusters, silloute_avgs_all_no_pca_hierarchical,  color=sns.color_palette()[0], marker='s',linestyle='--')
    a2.plot(clusters, siloute_numeric_no_pca_hierarchical,  color=sns.color_palette()[1], marker='s',linestyle='-')

    # a2.plot(clusters, siloute_avgs_all_no_pca, label='With categorical', color=sns.color_palette()[0], marker='o')
    # a2.plot(clusters, siloute_numeric_no_pca, label='Without categorical', color=sns.color_palette()[1],  marker='s')
    
    a1.set_title('PCA')
    a2.set_title('No PCA')
    scatters = [a1,a2]
    for ax in scatters:
        ax.set_xticks(clusters)
        ax.set_xticklabels(clusters)
        ax.set_xlabel('Clusters')
        ax.set_ylabel('Average Silhouette score')
        #ax.legend(loc='upper right', fontsize=8)
        ax.legend(handles=custom_lines,fontsize=8)
    siloute_avg_3,siloute_samples_3_all_pca,clusters_3_pca = siloute_all_pca_gmm[1]
    siloute_avg_3_no_pca,siloute_samples_3_all_no_pca,clusters_3_no_pca = siloute_all_no_pca_gmm[1]
    
    plot_dual_silhouettes_from_values(axes[8], siloute_samples_3_all_pca, siloute_samples_3_all_no_pca,
                                       clusters_3_pca, clusters_3_no_pca)
    if not os.path.exists('Figures'):
        try:
            os.makedirs('Figures')
        except OSError as e:
            print(f"Error creating directory: {e}")
    output_path = os.path.join('Figures', f'Fig_2.pdf')
        
    for i,ax in enumerate(axes):
        label = string.ascii_uppercase[i]
        ax.text(-0.1, 1.1, label, transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
    plt.tight_layout(pad=2)

    fig.savefig(output_path, format='pdf')


plot_figure_2()
