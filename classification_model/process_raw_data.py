""" This module contains one or several steps to process raw data.

These processing steps are independent of the real ML preprocessing steps, like filling NAs.
Therefore these steps are done before splitting the data.

"""
import pandas as pd
from typing import NoReturn, List

from util.logger import define_logger
from util.read_yaml import read_yaml

# Build logger
logger = define_logger()


def process_raw_data(raw_df: pd.DataFrame, cols_to_be_removed: List[str]) -> NoReturn:
    """Processes the raw input data.

    Args:
        raw_df (pd.DataFrame): Raw dataframe to be processed
        cols_to_be_removed (List[str]): List containing column names to be removed

    Returns:
        None

    """

    # Preprocess data
    raw_df_processed = raw_df.drop(columns=cols_to_be_removed)

    return raw_df_processed


if __name__ == "__main__":

    logger.info("Raw input data processing started ...")

    # Get classification model config yaml file
    clf_model_conf = read_yaml("classification_model_config.yaml")

    # Get information from config yaml file
    cols_to_be_removed = clf_model_conf["preprocess_data"]["cols_to_be_removed"]
    input_data_path = clf_model_conf["preprocess_data"]["paths"]["data_in_path"]
    output_data_path = clf_model_conf["preprocess_data"]["paths"]["data_in_path"]

    # Read input dataframe
    raw_df = pd.read_csv(input_data_path)

    # Process raw data
    raw_df_processed = process_raw_data(
        raw_df=raw_df, cols_to_be_removed=cols_to_be_removed
    )

    # Write dataframe to output folder
    raw_df_processed.to_csv(output_data_path, index=False)

    logger.info("Raw input data processing finished!")
