from enum import Enum
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

from prince import MCA

from plotting_utils import plot_dim_reduction,sub_plot_dim


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
    'TTH_MSS_YN': FeatureType.CATEGORY,
    'ODT_TRB_YN': FeatureType.CATEGORY,
    'WSDM_DIS_YN': FeatureType.CATEGORY,


    # 'DATE':,
}
dental_features = ['HCHK_CE_IN','CRS_YN','TTH_MSS_YN','ODT_TRB_YN','WSDM_DIS_YN','TTR_YN']
features_to_remove = ['DATE', 'YEAR', 'IDV_ID']
def get_features_in_data(columns, global_features):
    feature_in_data = {feature: global_features[feature] for feature in columns if feature in global_features}
    return feature_in_data

def convert_category_string(data, features_in_data):
    
    convert_dict = {'Y': 1, 'N': 0}
    for (feature_name, feature_type) in features_in_data.items():
        if feature_type == FeatureType.CATEGORY:
            data[feature_name].replace(convert_dict, inplace=True)
    return data


def impute_missing_values(data, features_in_data):
    '''
    Impute missing values:
        for numeric features: impute with the mean value
        for categorical features: impute with the median value'''
    print("Impute missing values with the median or average value.")
    for (feature_name, feature_type) in features_in_data.items():
        
        if feature_type == FeatureType.NUMBER:
            data.loc[(data[feature_name].isnull() == True), feature_name] = data[
                feature_name].mean()

        if feature_type == FeatureType.CATEGORY:
            data.loc[(data[feature_name].isnull() == True), feature_name] = data[
                feature_name].median()
    return data


def remove_features_from_data(data, feature_to_remove):
    
    try:
        return data.drop(columns=feature_to_remove)
    except KeyError as e:
        print(f"Error: {e} not found in the data frame")
        return data

def remove_features_from_global_features(features_to_remove):
    if isinstance(features_to_remove, str):
        features.pop(features_to_remove, None)
    elif isinstance(features_to_remove, list):
        for feature in features_to_remove:
            features.pop(feature, None)
def dealing_with_categorical_features(data, categorical_features, numeric_features):
    '''
    Takes a data with and reduced using MCA its categorical features
    Returns a combined data frame with MCA components and numeric features
    Args:
        data: Data frame
        categorical_features (list): List of categorical features
        numeric_features (list): List of numeric features
    Returns:
        combined_data (pd.DataFrame): Combined data frame with MCA components and numeric features'''
    encoder = OneHotEncoder(sparse_output=False)
    encoded_categorical = encoder.fit_transform(data[categorical_features])
    encoded_categorical_df = pd.DataFrame(encoded_categorical,
                                          columns=encoder.get_feature_names_out(categorical_features),index=data.index)

    mca_reduced_data = reduce_dimensions_mca(encoded_categorical_df, n_components=len(categorical_features))
    mca_reduced_data.columns = [str(col) for col in mca_reduced_data.columns]
    # Combine encoded categorical features with numeric features
    combined_data = pd.concat([mca_reduced_data, data[numeric_features]], axis=1)

    return combined_data

def dim_reduction(data, n_components, method):
    if method.lower() == 'pca':
        return reduce_dimensions_pca(data, n_components)
    elif method.lower() == 'tsne':
        return reduce_dimensions_tsne(data, n_components)
    elif method.lower() == 'mca':
        return reduce_dimensions_mca(data, n_components)
    else:
        raise ValueError(f"Error: Method {method} not supported")

def reduce_dimensions_pca(data, n_components):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data)
    pca_df = pd.DataFrame(
        data=principal_components,
        columns=[f'PC{i + 1}' for i in range(n_components)],index=data.index if isinstance(data.index,pd.Index) else None
    )

    # print("\nExplained variance ratio:", pca.explained_variance_ratio_)
    # print("PCA results preview:")
    # print(pca_df.head())
    return pca_df
def reduce_dimensions_tsne(data, n_components):
    tsne = TSNE(n_components=n_components, perplexity=30, random_state=42, n_iter=1000)
    tsne_results = tsne.fit_transform(data)

    # Create a DataFrame for the t-SNE results
    tsne_df = pd.DataFrame(
        data=tsne_results,
        columns=[f'Dim{i + 1}' for i in range(n_components)],index=data.index if isinstance(data.index,pd.Index) else None
    )
    return tsne_df

def reduce_dimensions_mca(data, n_components):
    mca = MCA(n_components=n_components)
    reduced_data = mca.fit_transform(data)

    return reduced_data


def load_data(file_path):
    data = pd.read_csv(file_path)
    # print("Data preview:")
    # print(data.head())
    return data

def split_data_from_target(data, target_column= None):
    '''
    Remove the target column from the data and return the data, target
    Args:
        data (pd.DataFrame): Data frame
        target_column (str,list): Name of the target column
    Returns:
        data (pd.DataFrame): Data frame without the target column
        target (pd.Series): Target column'''
    if target_column is None:
        return data, None
    if isinstance(target_column,str):
        target_column = [target_column]
    # .reset_index(drop=True)
    target = data[target_column]
    data = data.drop(columns=target_column)
    return data, target

def add_target_data(data, target_data, target_column):
    '''
    Add the target column to the data and return the data
    Args:
        data (pd.DataFrame): Data frame
        target_data (pd.Series): Series with the target data
        target_column (str): Name of the target column
    Returns:
        data (pd.DataFrame): Data frame with the target column'''
    data[target_column] = target_data
    return data

def weighted_sample(df, frac, weight_column = None, random_state=42):
    """
    Sample a DataFrame while preserving the distribution of a given column.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        frac (float): Fraction of rows to sample (0 < frac ≤ 1).
        weight_column (list,str): Column name to use for weighting. if list of column given, merge the content to a singel column
        random_state (int, optional): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Sampled DataFrame maintaining the ratio of `weight_column`.
        column_name (str): Name of the column used for weighting.
    """
    assert 0 < frac <= 1, 'frac must be in the range (0, 1]'
    if weight_column is None or weight_column not in df.columns:
        train_df, sampled_data = train_test_split(df, test_size=frac, random_state=random_state)
        return sampled_data, None

    column_name = weight_column
    if isinstance(weight_column, list): # join values to a single column
        column_name = "_".join(weight_column)
        df[column_name] = df[weight_column].astype(str).agg("_".join, axis=1)
    counts = df[column_name].value_counts(normalize=True)
    train_df, sampled_data = train_test_split(df, test_size=frac, stratify=df[column_name], random_state=random_state)
    sk_counts = sampled_data[column_name].value_counts(normalize=True)
    for ori_count,sk_count in zip(counts,sk_counts):
        print(f'Original ratio in df: {ori_count:.4f}, After sampled ratio: {sk_count:.4f}')

    return sampled_data, column_name
def reduce_dimenseions_multiple_datasets(data_folder, n_components, method, target_column, features_to_remove):
    """
    Reduce dimensions of multiple datasets. 
    Plots all in one figure.
    """
    if isinstance(data_folder,dict):
        titles = list(data_folder.keys())
        data_frames = list(data_folder.values())
    else:
            
        data_frames = [load_data(os.path.join(data_folder,file)) for file in os.listdir(data_folder)]
        titles = [file.split('.')[0] for file in os.listdir(data_folder)]
    processed_data = [preprocess_data_for_dim(data, target_column, features_to_remove) for data in data_frames]
    reduced_dataframes = [(dim_reduction(data, n_components, method),target_data,target_column_name ) 
                          for data,target_data,target_column_name in processed_data]
    
    reduced_dataframes = [
    (add_target_data(data, target_data, target_column_name),target_column_name) if target_data is not None else (data,None)
    for data, target_data, target_column_name in reduced_dataframes
]
    if target_column is None:
        target_column = ""
    sub_plot_dim(reduced_dataframes,titles,generall_title=f'All datasets {method} {target_column}',method=method)
    

def preprocess_data_for_dim(data, target_column, features_to_remove):
    """
    Preprocess data for dimensionality reduction.

    1. Remove unwanted features.
    2. Impute missing values.
    3. Sample data. (assign target, optional)
    4. MCA categorial features.
    5. Standardize data.
    
    Returns:
        processed_data (pd.DataFrame): Processed data.
        target_data, target_column (pd.Series, str, optional): Target data and column name.
    """
    data = remove_features_from_data(data, features_to_remove)
    
    features_in_data = get_features_in_data(data.columns, features)
    data = convert_category_string(data, features_in_data)
    data = impute_missing_values(data, features_in_data)
    data,joined_column = weighted_sample(data, 0.03, target_column, random_state=42)
    if joined_column is not None:
        data,target = split_data_from_target(data, joined_column)
        data = remove_features_from_data(data, target_column)
        features_in_data.pop(joined_column,None)
    else: target = None
    
    categorical_features = [_name for (_name, _type) in features_in_data.items() if _type == FeatureType.CATEGORY]
    numeric_features = [_name for (_name, _type) in features_in_data.items() if _type == FeatureType.NUMBER]
    processed_data = dealing_with_categorical_features(data, categorical_features, numeric_features)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(processed_data)
    scaled_df = pd.DataFrame(data_scaled, columns=processed_data.columns, index=processed_data.index)

    return scaled_df, target, joined_column
def merge_column_values(data, column, range):
    
    
    data[column] = (data[column] - 1) // 2 + 1
    return data

def main():
    # data = load_data('/home/alon/Unsupervised learning/archive/2020.csv')
    # data = merge_column_values(data,'AGE_GROUP',None)
    # split_dfs = {name: group for name, group in data.groupby('AGE_GROUP')}
    # reduce_dimenseions_multiple_datasets(split_dfs,2,'pca',None,features_to_remove)
    reduce_dimenseions_multiple_datasets('/home/alon/Unsupervised learning/archive', 2, 'tsne', 'AGE_GROUP', features_to_remove)

    # data = load_data('/home/alon/Unsupervised learning/archive/2020.csv')
    # target_column = 'SMK_STAT' # 
    # # Preprocess data
    # data = remove_features_from_data(data, features_to_remove)
    # data = impute_missing_values(data)
    # data,joined_column = weighted_sample(data, 0.03, target_column, random_state=42)
    # data,target = split_data_from_target(data, joined_column)
    # data = remove_features_from_data(data, target_column)
    # remove_features_from_global_features(target_column)
    
    # categorical_features = [_name for (_name, _type) in features.items() if _type == FeatureType.CATEGORY]
    # numeric_features = [_name for (_name, _type) in features.items() if _type == FeatureType.NUMBER]
    # processed_data = dealing_with_categorical_features(data, categorical_features, numeric_features)

    # scaler = StandardScaler()
    # data_scaled = scaler.fit_transform(processed_data)
    # scaled_df = pd.DataFrame(data_scaled, columns=processed_data.columns, index=processed_data.index)

    # # Reduce dimensions using PCA
    # pca_reduced_data = reduce_dimensions_tsne(scaled_df, n_components=2)
    # pca_reduced_data = add_target_data(pca_reduced_data, target, joined_column)
    
    # plot_dim_reduction(pca_reduced_data, joined_column, save_fig=True, fig_name=joined_column, method='TSNE')


if __name__ == '__main__':
    main()
