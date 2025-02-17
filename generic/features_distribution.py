from plotting_utils import *


def main():
    """
    Main function to plot correlation matrix and features distribution.
    """
    data = pd.read_csv("..\data sets\Health Checkup Result\\2020.csv")
    data = data.drop(columns=["DATE"])
    plot_correlation_matrix(data)
    columns = data.columns
    plot_numeric_features_distribution(data, columns)
    do_boxplot(data)


if __name__ == '__main__':
    main()
