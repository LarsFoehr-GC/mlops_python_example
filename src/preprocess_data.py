""" This module is only for demonstration.

Therefore only one preprocessing step is done: One variable is removed -> SkinThickness
This is only for demonstration. For other data sets more preprocessing would be necessary.

"""

import logging
import os
import pandas as pd
import sys
import typing

from util.logger import define_logger

# Build logger
logging.basicConfig(
    format="%(asctime)s; %(levelname)s - %(name)s - %(message)s", level=logging.DEBUG
)
logger = logging.getLogger(__name__)


def preprocess_data(DATA_PATH: str) -> typing.NoReturn:
    """Preprocesses the original input data and writes two dataframes to the DATA_PATH.

    The first dataframe to be written out, contains the features used for modeling.
    The second dataframe to be written out, is the whole preprocessed dataset.

    Args:
        DATA_PATH (str): The path to the input data.

    Returns:
        None

    """

    logger.info("Data preprocessing started ...")

    # Read input dataframe
    df = pd.read_csv(DATA_PATH)

    # Preprocess data
    df_preprocessed = df.drop(columns=["SkinThickness"])

    # Write dataframe to folder
    df_preprocessed.to_csv("./data/diabetes_preprocessed.csv", index=False)

    logger.info("Data preprocessing finished!")


if __name__ == "__main__":
    DATA_PATH = os.path.abspath(sys.argv[1])
    preprocess_data(DATA_PATH)
