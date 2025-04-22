import os
from clustering import *
from data_features import FeatureType
from plotting_utils import *
from preprocessing import *
import string

fig = plt.figure(figsize=(9,12))

    
# Subplot layout (manual with GridSpec for flexibility)
import matplotlib.gridspec as gridspec

gs = gridspec.GridSpec(2, 3, figure=fig)  # 3 rows, 3 columns

# 9 box plots 3x3 pictures.
axes = []
axes.append(fig.add_subplot(gs[0, 0]))  # A
axes.append(fig.add_subplot(gs[0, 1]))    # B
axes.append(fig.add_subplot(gs[0, 2]))    # C

# Second row (3 subplots)
axes.append(fig.add_subplot(gs[1, 0]))    # D
axes.append(fig.add_subplot(gs[1, 1]))    # E
axes.append(fig.add_subplot(gs[1, 2]))    # F

cluster_method = 'gmm'
#cluster_method = 'hierarchical'
def plot_figure_2():
    data,features_in_data = get_working_data()

    remote_job_work = ['Remote_Work','Job_Level','Work_Life_Balance']
    data_no_remotejobwork = data.drop(columns=remote_job_work)
    feature_in_data_no_remotejobwork = features_in_data.copy()
    feature_in_data_no_remotejobwork = {k: v for k, v in feature_in_data_no_remotejobwork.items() if k not in remote_job_work}

    processed_data_no_remotejobwork,t,tc_ = preprocess_data_for_dim(data=data_no_remotejobwork,target_column=None,data_features=feature_in_data_no_remotejobwork)

    cluster_labels = cluster(data=processed_data_no_remotejobwork, n_clusters=3,method=cluster_method)
    for feature,ax in zip(remote_job_work,axes[:3]):
        labels = data[feature]
        merged_data = pd.DataFrame({'cluster_labels': cluster_labels, 'other_labels': labels})
        cross_tab_by_cluster = pd.crosstab(merged_data['cluster_labels'], merged_data['other_labels'], normalize='index')
        plot_cluster_crosstab(df=cross_tab_by_cluster,ax=ax,title=feature)
    


    processed_data,t,tc_ = preprocess_data_for_dim(data=data,target_column=None,data_features=features_in_data)
    reduced_data_pca = dim_reduction(processed_data, n_components=2, method='pca')

    #evaluate silloute
    categorical_numeric = split_data_into_categorical_and_numeric(data,features_in_data)
    numeric_df, numeric_features = categorical_numeric[1]
    numeric_features = {feature: FeatureType.NUMBER for feature in numeric_features}
    numeric_df,t,tc_ = preprocess_data_for_dim(data=numeric_df, target_column=None, data_features=numeric_features)
    numeric_pca = dim_reduction(numeric_df, n_components=2, method='pca')
    clusters = [i for i in range (2,10)]
    siloute_all_pca,siloute_numeric_pca,siloute_all_no_pca,siloute_numeric_no_pca = [],[],[],[]
    for i in clusters:
        siloute_all_pca.append(get_avg_siloute_score(X=reduced_data_pca, n_clusters=i,method=cluster_method))
        siloute_numeric_pca.append(get_avg_siloute_score(X=numeric_pca, n_clusters=i,method=cluster_method)[0])
        siloute_all_no_pca.append(get_avg_siloute_score(X=processed_data, n_clusters=i,method=cluster_method))
        siloute_numeric_no_pca.append(get_avg_siloute_score(X=numeric_df, n_clusters=i,method=cluster_method)[0])
    silloute_avgs_all_pca = [item[0] for item in siloute_all_pca]
    siloute_avgs_all_no_pca = [item[0] for item in siloute_all_no_pca]
    a1,a2 = axes[3],axes[4]
    a1.plot(clusters, silloute_avgs_all_pca, label='With categorical', color=sns.color_palette()[0], marker='o')
    a1.plot(clusters, siloute_numeric_pca, label='Without categorical', color=sns.color_palette()[1], marker='s')
    a2.plot(clusters, siloute_avgs_all_no_pca, label='With categorical', color=sns.color_palette()[0], marker='o')
    a2.plot(clusters, siloute_numeric_no_pca, label='Without categorical', color=sns.color_palette()[1],  marker='s')
    a1.set_title('With PCA')
    a2.set_title('Without PCA')
    scatters = [a1,a2]
    for ax in scatters:
        ax.set_xticks(clusters)
        ax.set_xticklabels(clusters)
        ax.set_xlabel('Clusters')
        ax.set_ylabel('Average Silhouette score')
        ax.legend(loc='upper right', fontsize=8)
    siloute_avg_3,siloute_samples_3_all_pca,clusters_3_pca = siloute_all_pca[1]
    siloute_avg_3_no_pca,siloute_samples_3_all_no_pca,clusters_3_no_pca = siloute_all_no_pca[1]
    
    plot_dual_silhouettes_from_values(axes[5], siloute_samples_3_all_pca, siloute_samples_3_all_no_pca,
                                       clusters_3_pca, clusters_3_no_pca)
    if not os.path.exists('Figures'):
        try:
            os.makedirs('Figures')
        except OSError as e:
            print(f"Error creating directory: {e}")
    output_path = os.path.join('Figures', f'Fig_2_{cluster_method}.pdf')
        
    for i,ax in enumerate(axes):
        label = string.ascii_uppercase[i]
        ax.text(-0.1, 1.1, label, transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
    plt.tight_layout(pad=2)

    fig.savefig(output_path, format='pdf')






    

    
plot_figure_2()