""" This module contains data splitting and preprocessing.

"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Tuple

from util.logger import define_logger
from util.read_yaml import read_yaml

logger = define_logger()


def get_variables(data: pd.DataFrame, y_feature: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Seperate the target variable from the feature matrix.

    y is the targed variable.
    X is the feature matrix.

    Args:
        data (pd.DataFrame): The dataframe to be worked with.
        y_feature (str): Column to be the y feature.

    Returns:
        X, y: Features matrix and targed variable.

    """
    X = data.drop([y_feature], axis=1)
    y = data[y_feature]

    return X, y


def preprocess_data(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[np.array, np.array]:
    """Preprocess the already processed raw data, to make it ready for the ML model.

    Args:
        X_train (pd.DataFrame): Training Set to be transformed
        X_test (pd.DataFrame): Test Set to be transformed

    Returns:
        X_train, X_test: Both transformed datasets as np.arrays
    """

    # Build preprocess Pipeline
    preprocess_pipe = Pipeline(
        [
            ("Mean_Imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
            ("Standard_Scaler", StandardScaler()),
        ]
    )

    # Use the pipeline on train and test set
    X_train = preprocess_pipe.fit_transform(X_train)
    X_test = preprocess_pipe.transform(X_test)

    return X_train, X_test


if __name__ == "__main__":

    logger.error("Data splitting and preprocessing started ...")

    # Get classification model config yaml file
    clf_model_conf = read_yaml("classification_model_config.yaml")

    # Get information from config yaml file
    y_feature = clf_model_conf["split_preprocess"]["y_feature"]
    input_data_path = clf_model_conf["split_preprocess"]["paths"]["data_in_path"]
    X_train_out_path = clf_model_conf["split_preprocess"]["paths"]["X_train_out_path"]
    X_test_out_path = clf_model_conf["split_preprocess"]["paths"]["X_test_out_path"]
    y_train_out_path = clf_model_conf["split_preprocess"]["paths"]["y_train_out_path"]
    y_test_out_path = clf_model_conf["split_preprocess"]["paths"]["y_test_out_path"]

    # Get diabetes data set
    diabetes_df = pd.read_csv(input_data_path)

    # Separate X and y
    X, y = get_variables(data=diabetes_df, y_feature=y_feature)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    # Transform training and test data
    X_train, X_test = preprocess_data(X_train=X_train, X_test=X_test)

    # Write data out
    np.savetxt(X_train_out_path, X_train, delimiter=",")
    np.savetxt(X_test_out_path, X_test, delimiter=",")
    y_train.to_csv(y_train_out_path, index=False)
    y_test.to_csv(y_test_out_path, index=False)

    logger.error("Data splitting and preprocessing finished!")
