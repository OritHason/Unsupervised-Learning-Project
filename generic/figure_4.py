import os
from clustering import *
from plotting_utils import *
from preprocessing import *
import string

#cluster_method = 'hierarchical'
cluster_method = 'gmm'

def plot_pca_feature_contribution(ax=None):
    ## get working data blala
    data,features_in_data = get_working_data()
    data = data.drop(columns=['Remote_Work'])
    features_in_data.pop('Remote_Work')
    processed,t_,tc_ = preprocess_data_for_dim(data=data, target_column=None,
                                               data_features= features_in_data
                                               )
    pca = reduce_dimensions_pca(processed, n_components=2,return_pca=True)
    visualise_feature_coeffiecnts_pca(loadings=None,data=processed,pca=pca,ax=ax)


fig = plt.figure(figsize=(9,9))

# Subplot layout (manual with GridSpec for flexibility)
import matplotlib.gridspec as gridspec

gs = gridspec.GridSpec(4, 4, figure=fig)  # 3 rows, 3 columns
axes = []
axes.append(fig.add_subplot(gs[0:2, 0:2]))  # A
axes.append(fig.add_subplot(gs[0:2, 2:]))    # B
axes.append(fig.add_subplot(gs[2:, 0:2]))    # C
axes.append(fig.add_subplot(gs[2:, 2:]))    # D





def plot_figure_4():
    
    args = analyze_feature_diffriniation_both_cato_no_cato(remove_remote_work=True,cluster_method=cluster_method)
    dim_reduced_data_numeric = args[1]
    dim_reduced_data_numeric.drop(columns=['PC1','PC2'],inplace=True)
    clusters = dim_reduced_data_numeric.groupby('cluster_labels')
    for i, (ax, (cluster_value, cluster_data)) in enumerate(zip(axes[:3], clusters)):
        #show_ticks = (i == 2)  # Show ticks only for the second subplot (middle)
        show_ticks = True
        cluster_data = cluster_data.drop(columns=['cluster_labels'])
        plot_correlation_matrix(data=cluster_data, ax=ax, ax_title=f'Cluster {cluster_value}', show_ticks=show_ticks)
    plot_pca_feature_contribution(ax=axes[3])
    
    if not os.path.exists('Figures'):
        try:
            os.makedirs('Figures')
        except OSError as e:
            print(f"Error creating directory: {e}")
    output_path = os.path.join('Figures', f'Fig_4_{cluster_method}.pdf')



    for i,ax in enumerate(axes):
        label = string.ascii_uppercase[i]
        ax.text(-0.1, 1.1, label, transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
    plt.tight_layout(pad=0.5)
    fig.savefig(output_path, format='pdf')



plot_figure_4()