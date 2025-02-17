from enum import Enum
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from prince import MCA

from generic.plotting_utils import plot_pca_2d


class FeatureType(Enum):
    NUMBER = 0,
    CATEGORY = 1,


features = {
    # 'YEAR': ,
    # 'IDV_ID': ,
    'AREA_CODE': FeatureType.CATEGORY,
    'SEX': FeatureType.CATEGORY,
    'AGE_GROUP': FeatureType.CATEGORY,
    'HEIGHT': FeatureType.NUMBER,
    'WEIGHT': FeatureType.NUMBER,
    'WAIST': FeatureType.NUMBER,
    'SIGHT_LEFT': FeatureType.NUMBER,
    'SIGHT_RIGHT': FeatureType.NUMBER,
    'HEAR_LEFT': FeatureType.NUMBER,
    'HEAR_RIGHT': FeatureType.NUMBER,
    'BP_HIGH': FeatureType.NUMBER,
    'BP_LWST': FeatureType.NUMBER,
    'BLDS': FeatureType.NUMBER,
    'TOT_CHOLE': FeatureType.NUMBER,
    'TRIGLYCERIDE': FeatureType.NUMBER,
    'HDL_CHOLE': FeatureType.NUMBER,
    'LDL_CHOLE': FeatureType.NUMBER,
    'HMG': FeatureType.NUMBER,
    'OLIG_PROTE_CD': FeatureType.CATEGORY,
    'CREATININE': FeatureType.NUMBER,
    'SGOT_AST': FeatureType.NUMBER,
    'SGPT_ALT': FeatureType.NUMBER,
    'GAMMA_GTP': FeatureType.NUMBER,
    'SMK_STAT': FeatureType.CATEGORY,
    'DRK_YN': FeatureType.CATEGORY,
    'HCHK_CE_IN': FeatureType.CATEGORY,
    'CRS_YN': FeatureType.CATEGORY,
    'TTR_YN': FeatureType.CATEGORY,
    # 'DATE':,
}
features_to_remove = ['DATE', 'YEAR', 'IDV_ID']


def impute_missing_values(data):
    print("Impute missing values with the median or average value.")
    for (feature_name, feature_type) in features.items():
        if feature_type == FeatureType.NUMBER:
            data.loc[(data[feature_name].isnull() == True), feature_name] = data[
                feature_name].mean()

        if feature_type == FeatureType.CATEGORY:
            data.loc[(data[feature_name].isnull() == True), feature_name] = data[
                feature_name].median()
    return data


def remove_features(data, feature_to_remove):
    return data.drop(columns=feature_to_remove)


def dealing_with_categorical_features(data, categorical_features, numeric_features):
    encoder = OneHotEncoder(sparse_output=False)
    encoded_categorical = encoder.fit_transform(data[categorical_features])
    encoded_categorical_df = pd.DataFrame(encoded_categorical,
                                          columns=encoder.get_feature_names_out(categorical_features))

    mca_reduced_data = reduce_dimensions_mca(encoded_categorical_df, n_components=len(categorical_features))
    mca_reduced_data.columns = [str(col) for col in mca_reduced_data.columns]
    # Combine encoded categorical features with numeric features
    combined_data = pd.concat([mca_reduced_data, data[numeric_features]], axis=1)

    return combined_data


def reduce_dimensions_pca(data, n_components):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data)
    pca_df = pd.DataFrame(
        data=principal_components,
        columns=[f'PC{i + 1}' for i in range(n_components)]
    )

    # print("\nExplained variance ratio:", pca.explained_variance_ratio_)
    # print("PCA results preview:")
    # print(pca_df.head())
    return pca_df


def reduce_dimensions_mca(data, n_components):
    mca = MCA(n_components=n_components)
    reduced_data = mca.fit_transform(data)

    return reduced_data


def load_data(file_path):
    data = pd.read_csv(file_path)
    # print("Data preview:")
    # print(data.head())
    return data


def main():
    data = load_data('../data sets/Health Checkup Result/2020.csv')

    # Preprocess data
    data = remove_features(data, features_to_remove)
    data = impute_missing_values(data)
    categorical_features = [_name for (_name, _type) in features.items() if _type == FeatureType.CATEGORY]
    numeric_features = [_name for (_name, _type) in features.items() if _type == FeatureType.NUMBER]
    processed_data = dealing_with_categorical_features(data, categorical_features, numeric_features)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(processed_data)
    # Reduce dimensions using PCA
    pca_reduced_data = reduce_dimensions_pca(data_scaled, n_components=2)
    plot_pca_2d(pca_reduced_data)


if __name__ == '__main__':
    main()
