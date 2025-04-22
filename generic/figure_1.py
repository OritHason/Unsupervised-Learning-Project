from matplotlib import pyplot as plt

from preprocessing import *

from plotting_utils import plot_2d_scatter_on_ax
from clustering import gmm
import os


fig = plt.figure(figsize=(9, 10))

# Subplot layout (manual with GridSpec for flexibility)
import matplotlib.gridspec as gridspec

gs = gridspec.GridSpec(4, 3, figure=fig)  # 3 rows, 3 columns

# First row (2 subplots)
axes = []
axes.append(fig.add_subplot(gs[0, 0:2]))  # A
axes.append(fig.add_subplot(gs[0, 2]))    # B

# Second row (3 subplots)
axes.append(fig.add_subplot(gs[1, 0]))    # C
axes.append(fig.add_subplot(gs[1, 1]))    # D
axes.append(fig.add_subplot(gs[1, 2]))    # E

# Third row (3 subplots)
axes.append(fig.add_subplot(gs[2, 0:1]))  # F
axes.append(fig.add_subplot(gs[2, 1:2]))  # G
axes.append(fig.add_subplot(gs[2, 2]))    # H


# Fourth row (2 subplots)
axes.append(fig.add_subplot(gs[3, 0]))    # I
axes.append(fig.add_subplot(gs[3, 1]))    # J
axes.append(fig.add_subplot(gs[3, 2]))    # K
a1, a2, a3, a4, a5, a6, a7,a8,a9,a10,a11 = axes
def plot_figure_1():
    """
    1,1 - A pca
    1,2 - B tsne
    1,1 - C tsne remote work
    2,2 - D tsne work life balance
    2,3 - E tsne job level
    3,1 - F tsne no remote work work life balance
    3,2 - G tsne no remote work job level"""
    data,features_in_data = get_working_data()
    processed_data,t,tc_ = preprocess_data_for_dim(data=data,target_column=None,data_features=features_in_data)
    reduced_data_pca = dim_reduction(processed_data, n_components=2, method='pca')
    reduced_data_tsne_with_remote = dim_reduction(processed_data, n_components=2, method='tsne')
    plot_2d_scatter_on_ax(pca_df=reduced_data_pca,target_column=None,ax=a1,ax_title='PCA')
    plot_2d_scatter_on_ax(pca_df=reduced_data_tsne_with_remote,target_column=None,ax=a2,ax_title='t-SNE',method='tsne',prefix='Dim',explanation_prefix='Dimension')
    
    remote_data, remote_labels = convert_categorial_labels_into_numbers(data,'Remote_Work')
    work_life_data, work_life_labels = convert_categorial_labels_into_numbers(data,'Work_Life_Balance')
    job_level_data, job_level_labels = convert_categorial_labels_into_numbers(data,'Job_Level')
    reduced_data_tsne_with_remote = add_target_data(data=reduced_data_tsne_with_remote, target_data=remote_data, target_column='Remote_Work')
    reduced_data_tsne_with_remote = add_target_data(data=reduced_data_tsne_with_remote, target_data=work_life_data, target_column='Work_Life_Balance')
    reduced_data_tsne_with_remote = add_target_data(data=reduced_data_tsne_with_remote, target_data=job_level_data, target_column='Job_Level')
    reduced_data_pca_with_remote = add_target_data(data=reduced_data_pca, target_data=remote_data, target_column='Remote_Work')
    plot_2d_scatter_on_ax(pca_df=reduced_data_tsne_with_remote,target_column='Remote_Work',labels=remote_labels,ax=a3,ax_title='Remote work', method='tsne',prefix='Dim',explanation_prefix='Dimension')
    plot_2d_scatter_on_ax(pca_df=reduced_data_tsne_with_remote,target_column='Work_Life_Balance',labels=work_life_labels,ax=a4,ax_title='Work life balance',method='tsne',prefix='Dim',explanation_prefix='Dimension')
    plot_2d_scatter_on_ax(pca_df=reduced_data_tsne_with_remote,target_column='Job_Level',labels=job_level_labels,ax=a5,ax_title='Job level',method='tsne',prefix='Dim',explanation_prefix='Dimension')

    # # remove remote
    data_no_remote = data.drop(columns=['Remote_Work'])
    feature_in_data_no_remote = features_in_data.copy()

    feature_in_data_no_remote.pop('Remote_Work')
    processed_data_no_remote, t, tc_ = preprocess_data_for_dim(data=data_no_remote, target_column=None, data_features=feature_in_data_no_remote)
    reduced_data_tsne_without_remote = dim_reduction(processed_data_no_remote, n_components=2, method='tsne')
    work_life_data, work_life_labels = convert_categorial_labels_into_numbers(data_no_remote,'Work_Life_Balance')
    job_level_data, job_level_labels = convert_categorial_labels_into_numbers(data_no_remote,'Job_Level')
    reduced_data_tsne_without_remote = add_target_data(data=reduced_data_tsne_without_remote, target_data=work_life_data, target_column='Work_Life_Balance')
    reduced_data_tsne_without_remote = add_target_data(data=reduced_data_tsne_without_remote, target_data=job_level_data, target_column='Job_Level')
    plot_2d_scatter_on_ax(pca_df=reduced_data_tsne_without_remote, target_column='Work_Life_Balance', labels=work_life_labels, ax=a6, ax_title='Work life balance', method='tsne', prefix='Dim', explanation_prefix='Dimension')
    plot_2d_scatter_on_ax(pca_df=reduced_data_tsne_without_remote, target_column='Job_Level', labels=job_level_labels, ax=a7,ax_title='Job level', method='tsne', prefix='Dim', explanation_prefix='Dimension')
    plot_2d_scatter_on_ax(pca_df=reduced_data_pca_with_remote, target_column='Remote_Work', labels=remote_labels, ax=a8, ax_title='Remote work', method='pca')
    
    
    data_no_remote_no_job_level = data_no_remote.drop(columns=['Job_Level'])
    feature_in_data_no_remote_no_job_level = feature_in_data_no_remote.copy()
    feature_in_data_no_remote_no_job_level.pop('Job_Level')
    data_no_remote_no_work_life = data_no_remote.drop(columns=['Work_Life_Balance'])
    feature_in_data_no_remote_no_work_life = feature_in_data_no_remote.copy()
    feature_in_data_no_remote_no_work_life.pop('Work_Life_Balance')
    processed_data_no_remote_no_job_level,t,tc_ = preprocess_data_for_dim(data=data_no_remote_no_job_level,target_column=None,data_features=feature_in_data_no_remote_no_job_level)
    processed_data_no_remote_no_work_life,t,tc_ = preprocess_data_for_dim(data=data_no_remote_no_work_life,target_column=None,data_features=feature_in_data_no_remote_no_work_life)
    #processed_data_no_remote_no_job_level_no_work_life,t,tc_ = preprocess_data_for_dim(data=data_no_remote_no_job_level_no_work_life,target_column=None,data_features=feature_in_data_no_remote_no_job_level_no_work_life)
    datas = [processed_data,processed_data_no_remote_no_job_level,processed_data_no_remote_no_work_life]
    names = ['All features','No remote and\n job level','No remote and\n work life balance']
    labels = ['Remote_Work','Work_Life_Balance','Job_Level']
    
    for data_,name_,label_,ax in zip(datas,names,labels,axes[8:]):
        labels_ = data[label_]
        #cluster_labels_hirearchial = hirearchial_clustering(data=data_,n_clusters=3,only_labels=True)
        cluster_labels_gmm = gmm(data=data_,n_clusters=3)
        #merged_data_h = pd.DataFrame({'cluster_labels': cluster_labels_hirearchial, 'other_labels': labels_})
        merged_data_g = pd.DataFrame({'cluster_labels': cluster_labels_gmm, 'other_labels': labels_})
        #print(f'{name_}, hirechial:\n',pd.crosstab(merged_data_h['cluster_labels'], merged_data_h['other_labels'], normalize='index'))
        cross_tab=pd.crosstab(merged_data_g['cluster_labels'], merged_data_g['other_labels'], normalize='index')
        plot_cluster_crosstab(df=cross_tab,ax=ax,title=name_)

plot_figure_1()
fig.tight_layout(pad=2)
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H','I','J','K']
for ax, label in zip(axes, labels):
    ax.text(-0.25, 1.25, label, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='left')


out_dir = 'Figures'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

plt.savefig(f'{out_dir}/Fig_1.pdf', bbox_inches="tight", pad_inches=0.2)