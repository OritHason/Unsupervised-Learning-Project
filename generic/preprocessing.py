import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

from prince import MCA

from plotting_utils import *
from data_features import *


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
        if data[feature_name].isnull().sum() == 0:
            continue
        if feature_type == FeatureType.NUMBER:
            data.loc[(data[feature_name].isnull() == True), feature_name] = data[
                feature_name].mean()

        if feature_type == FeatureType.CATEGORY:

            data.loc[(data[feature_name].isnull() == True), feature_name] = data[
                feature_name].median()
    return data

def encode_categorial_to_onehot(data, categorical_features):
    """
    Encode categorial features (strings/ints) to one-hot encodings.
    """
    encoder = OneHotEncoder(sparse_output=False)
    encoded_categorical = encoder.fit_transform(data[categorical_features])
    encoded_categorical_df = pd.DataFrame(encoded_categorical,
                                          columns=encoder.get_feature_names_out(categorical_features),index=data.index)
    return encoded_categorical_df
def remove_features_from_data(data, feature_to_remove):
    if feature_to_remove is None:
        print("No features to remove")
        return data
    try:
        return data.drop(columns=feature_to_remove)
    except KeyError as e:
        print(f"Error: {e} not found in the data frame")
        return data


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
    if not categorical_features: # empty
        return data
    encoder = OneHotEncoder(sparse_output=False)
    encoded_categorical = encoder.fit_transform(data[categorical_features])
    encoded_categorical_df = pd.DataFrame(encoded_categorical,
                                          columns=encoder.get_feature_names_out(categorical_features),index=data.index)

    mca_reduced_data = reduce_dimensions_mca(encoded_categorical_df, n_components=len(categorical_features))
    mca_reduced_data.columns = [str(col) for col in mca_reduced_data.columns]
    # Combine encoded categorical features with numeric features
    combined_data = pd.concat([mca_reduced_data, data[numeric_features]], axis=1)

    return combined_data

def dim_reduction(data, n_components, method, **kargs):
    if method.lower() == 'pca':
        return reduce_dimensions_pca(data, n_components)
    elif method.lower() == 'tsne':
        return reduce_dimensions_tsne(data, n_components,**kargs)
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

    print("\nExplained variance ratio:", pca.explained_variance_ratio_)
    # print("PCA results preview:")
    # print(pca_df.head())
    return pca_df
def reduce_dimensions_tsne(data, n_components, perplexity = 30):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42, n_iter=1000)
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
    data_copied = data.copy()
    data_copied[target_column] = target_data
    return data_copied
    

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
    if frac == 1:
        return df, weight_column
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





def plot_catogerial_feature_for_reduced_data(data, data_features_types, target_column, method, 
                                             n_components, data_name):
    """
    Plot the data points by each category in the categorial feature.
    
    plot_catogerial_feature_for_reduced_data(data,features_in_data,['Remote_Work','Work_Life_Balance','Job_Level'],method,2,data_name)

    """
    data = load_data(data) if isinstance(data, str) else data
    if isinstance(target_column,str):
        target_column = [target_column]
    target_data_by_col,label_by_col = [], []
    for target_col in target_column:
        target_data,labels = convert_categorial_labels_into_numbers(data,target_col)
        target_data_by_col.append(target_data)
        label_by_col.append(labels)
    dim_data,t_,tc_=preprocess_data_for_dim(data,target_column=None,
                                                                data_features=data_features_types,remove_target=False,wieght_fraction=1)
    reduced_data = dim_reduction(dim_data, n_components=n_components, method=method)
    data_by_categoric, titles = [], []
    for target_data,labels in zip(target_data_by_col,label_by_col):
        for categoric_number,categoric in labels.items():
            categoric_indexes = target_data[target_data == categoric_number].index
            categoric_data = reduced_data.loc[categoric_indexes]
            data_by_categoric.append(categoric_data)
            titles.append(categoric)
    data_by_categoric.append(reduced_data)
    titles.append('all')
    sub_plot_dim_categoric(data_by_categoric,titles,generall_title=f'Columns: {target_column} {method} {data_name}',method=method)
    


def reduce_dim_for_multiple_targets(data, data_features,  targets,
                                    n_components, method, data_name, remove_targets = True, wieght_fraction = 1,
                                      exteranl_targets=None, external_data = None):
    """
    Reduce dimensions to the using reduction method and plot it on 2d.
    Plot the data points by each feature the target list.
    If remove_targets is True, remove the wanted targets from the data before the dim reduction.
    Plots by all targets in the same plots - each feature a subplot.
    
    Args:
        data (pd.DataFrame/path): Data frame.
        data_features (dict): Data features.
        data_features_to_remove (list): Features to remove.
        targets (list): List of target columns.
        n_components (int): Number of components.
        method (str): Method for dimensionality reduction.
    """
    
    data = load_data(data) if isinstance(data, str) else data
    if not remove_targets: # 1 data frame for all targets just paint the data differently
        dim_data,target_data,target_columns=preprocess_data_for_dim(data,target_column=None,
                                                                data_features=data_features,remove_target=False,wieght_fraction=wieght_fraction)
        all_targets_data_and_labels = []
        for target_col in targets:
            if len(data) != len(dim_data): # dim_data is sampled
                target_data = data[target_col].loc[dim_data.index]
            else:
                target_data = data[target_col]
            target_data,labels_col = convert_categorial_labels_into_numbers(target_data,target_col)
            all_targets_data_and_labels.append((target_data,target_col,labels_col))
        if exteranl_targets is not None:
            targets = targets + exteranl_targets
            for target_col in exteranl_targets:
                target_data = external_data[target_col]
                target_data,labels_col = convert_categorial_labels_into_numbers(target_data,target_col)
                all_targets_data_and_labels.append((target_data,target_col,labels_col))
        if method == 'tsne':
            perplexity = 10
        else: perplexity=None   
        reduced_data = dim_reduction(dim_data, n_components=2, method=method)
        all_reduced_data = [(add_target_data(reduced_data,target_data,target_column),target_column,labels_col) for target_data,target_column,labels_col in all_targets_data_and_labels]
        
        sub_plot_dim(all_reduced_data,targets,generall_title=f'All targets {method} {data_name} without cato',method=method)
        return

                                                                
    target_data,labels = convert_categorial_labels_into_numbers(target_data,target_col)
    reduced_data = dim_reduction(dim_data, n_components=2, method=method)
    reduced_data = add_target_data(reduced_data,target_data,target_columns)

    processed_data = [preprocess_data_for_dim(data, target_column, data_features, remove_target=False) for target_column in targets]
    reduced_dataframes = [(dim_reduction(data, n_components, method),target_data,target_column_name ) 
                          for data,target_data,target_column_name in processed_data]
    
    reduced_dataframes = [
    (add_target_data(data, target_data, target_column_name),target_column_name) if target_data is not None else (data,None)
    for data, target_data, target_column_name in reduced_dataframes
]
    sub_plot_dim(reduced_dataframes,targets,generall_title=f'All targets {method} {data_name}',method=method)


def preprocess_data_for_dim(data, target_column, data_features, 
                            wieght_fraction = 1, remove_target = True):
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
    features_in_data= data_features.copy()
    data = impute_missing_values(data, features_in_data)
    data,joined_column = weighted_sample(data, wieght_fraction, target_column, random_state=42)
    if joined_column is not None:
        if remove_target:
            data,target = split_data_from_target(data, joined_column)
            data = remove_features_from_data(data, target_column)
            features_in_data.pop(joined_column,None)
        else:
            target = data[joined_column]
            
    else: target = None
    
    categorical_features = [_name for (_name, _type) in features_in_data.items() if _type == FeatureType.CATEGORY]

    numeric_features = [_name for (_name, _type) in features_in_data.items() if _type == FeatureType.NUMBER]
    processed_data = dealing_with_categorical_features(data, categorical_features, numeric_features)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(processed_data)
    scaled_df = pd.DataFrame(data_scaled, columns=processed_data.columns, index=processed_data.index)

    return scaled_df, target, joined_column
def transform_age_to_categorial(data, age_column):
    """
    Transform age to categorial feature.
    """
    min_age = data[age_column].min()
    max_age = data[age_column].max()
    age_gap = 5
    bins = list(range(min_age, max_age + age_gap, age_gap))  
    labels = list(range(len(bins) - 1)) 

    data[age_column] = pd.cut(data[age_column], bins, labels=labels,include_lowest=True)

    return data
def convert_categorial_labels_into_numbers(data, column):
    """
    Create a mapping dict from categorial features into numbers and vs versa.
    """
    target_vals = data[column] if isinstance(data,pd.DataFrame) else data
    if pd.api.types.is_numeric_dtype(target_vals):
        return target_vals,None
    unique_vals = set(target_vals)
    labels = {val: i for i, val in enumerate(unique_vals)}
    target_vals = target_vals.map(labels)
    labels = {v: k for k, v in labels.items()}
    return target_vals,labels
def main_working_data(return_target_column = False):
    """
    Return the working data as a tuple of 2 data frames:
    (reduced dimension data, preprocessed data)
    """
    data_name = 'Working data'
    data,features_in_data = get_working_data()
    targets = data.columns.tolist()
    method = 'tsne'    
    
    
    targets_to_remove = ['Remote_Work','Work_Life_Balance','Job_Level']
    targets_data = data[targets_to_remove]
    data = remove_features_from_data(data, targets_to_remove)
    targets = [target for target in targets if target not in targets_to_remove]
    for target in targets_to_remove:
        if target in features_in_data:
            features_in_data.pop(target)
    reduce_dim_for_multiple_targets(data,features_in_data,targets,2,method,data_name,remove_targets=False,
                                    external_data=targets_data,exteranl_targets=targets_to_remove)
    
def main_korea_data():
    data_name = 'korea_data'
    data = load_data("/home/alon/Unsupervised learning/archive/2020.csv")
    data = remove_features_from_data(data,korea_features_to_remove)
    method = 'pca'
    features_in_data = get_features_in_data(data.columns, korea_features)
    targets = data.columns.tolist()
    reduce_dim_for_multiple_targets(data,features_in_data,targets,2,method,data_name,remove_targets=False, wieght_fraction=0.03)
 
def get_working_data():
    data = load_data("/home/alon/Unsupervised learning/Unsupervised-Learning-Project/Datasets/corporate_work_hours_productivity.csv")
    data = remove_features_from_data(data,work_features_to_remove)
    features_in_data = get_features_in_data(data.columns, work_features)
    data = transform_age_to_categorial(data, 'Age')
    features_in_data['Age'] = FeatureType.CATEGORY
    return data,features_in_data

if __name__ == '__main__':
    main_working_data()
