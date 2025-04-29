import string

from generic.clustering import *
from generic.preprocessing import *


def plot_figure_3():
    cluster_method = 'gmm'
    # cluster_method = 'hierarchical'
    feature_order_numeric = ['Meetings_per_Week', 'Job_Satisfaction', 'Tasks_Completed_Per_Day',
                             'Productivity_Score', 'Annual_Salary', 'Absences_Per_Year',
                             'Monthly_Hours_Worked', 'Overtime_Hours_Per_Week', 'Years_at_Company'
                             ]

    categorical_anova_path = os.path.join(os.getcwd(), 'generic', 'Statistics',
                                          f'Working data_with_categorical_reduced_{cluster_method}_anova_results.csv')
    without_categorical_anova_path = os.path.join(os.getcwd(), 'generic', 'Statistics',
                                                  f'Working data_without_categorical_reduced_{cluster_method}_anova_results.csv')
    statistics_path = os.path.join(os.path.dirname(__file__), 'Statistics')
    if not os.path.exists(statistics_path):
        try:
            os.makedirs(statistics_path)
        except OSError as e:
            print(f"Error creating directory: {e}")
    fig = plt.figure(figsize=(9, 12))

    # Subplot layout (manual with GridSpec for flexibility)
    import matplotlib.gridspec as gridspec

    gs = gridspec.GridSpec(3, 3, figure=fig)  # 3 rows, 3 columns

    # 9 box plots 3x3 pictures.
    axes = []
    axes.append(fig.add_subplot(gs[0, 0]))  # A
    axes.append(fig.add_subplot(gs[0, 1]))  # B
    axes.append(fig.add_subplot(gs[0, 2]))  # C

    # Second row (3 subplots)
    axes.append(fig.add_subplot(gs[1, 0]))  # D
    axes.append(fig.add_subplot(gs[1, 1]))  # E
    axes.append(fig.add_subplot(gs[1, 2]))  # F

    # Third row (3 subplots)
    axes.append(fig.add_subplot(gs[2, 0]))  # G
    axes.append(fig.add_subplot(gs[2, 1]))  # H
    axes.append(fig.add_subplot(gs[2, 2]))  # I

    if not os.path.exists(categorical_anova_path):
        print("Anova results files are not found for all features. Running anova analysis first.")
        analyze_feature_diffrintiation_per_cluster(cluster_method=cluster_method,num_clusters=3,
                                            data_name='Working data',remove_categorical=False,remove_remote_work=True)
    if not os.path.exists(without_categorical_anova_path):
        print("Anova results files are not found for numeric features. Running anova analysis first.")
        analyze_feature_diffrintiation_per_cluster(cluster_method=cluster_method,num_clusters=3,
                                            data_name='Working data',remove_categorical=True,remove_remote_work=True)
    args = analyze_feature_diffriniation_both_cato_no_cato(categorical_anova_path=categorical_anova_path,
                                                            no_categorical_anova_path=without_categorical_anova_path,
                                                            remove_remote_work=True,cluster_method=cluster_method)
    features_order = feature_order_numeric
    if not os.path.exists('Figures'):
        try:
            os.makedirs('Figures')
        except OSError as e:
            print(f"Error creating directory: {e}")
    output_path = os.path.join('Figures', f'Fig_3_{cluster_method}.pdf')
    plot_box_cluster_together(*args,feature_order=features_order,axes=axes,fig=fig)

    for i,ax in enumerate(axes):
        label = string.ascii_uppercase[i]
        ax.text(-0.1, 1.1, label, transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

    fig.savefig(output_path, format='pdf')


# plot_figure_3()
