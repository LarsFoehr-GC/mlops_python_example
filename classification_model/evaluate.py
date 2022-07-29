""" This module contrains the model evaluation.

Some metrics are calculated and saved for train and test data.

"""
import json
import numpy as np
from numpy import genfromtxt
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from typing import Dict

from util.logger import define_logger
from util.read_yaml import read_yaml

logger = define_logger()


def comb_eval(y: np.array, y_pred: np.array) -> Dict:
    """Calculate several KPIs to evaluate the model.
    Args:
        y (np.array): Training target variable.
        y_pred (np.array): Test targed variable.
    Returns:
        comb_eval_dict: Dictionary containing several evaluation KPIs.
    """
    acc = accuracy_score(y, y_pred)
    recall = recall_score(y, y_pred)
    precision = precision_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    comb_eval_dict = {
        "accuracy": acc,
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }

    return comb_eval_dict


if __name__ == "__main__":

    logger.error("Model evaluation started ...")

    # Get classification model config yaml file
    clf_model_conf = read_yaml("classification_model_config.yaml")

    # Load the necessary data
    y_train = genfromtxt(
        clf_model_conf["evaluate"]["paths"]["y_train_in_path"],
        delimiter=",",
        skip_header=1,
    )
    y_pred_train = genfromtxt(
        clf_model_conf["evaluate"]["paths"]["y_pred_train_in_path"], delimiter=","
    )
    y_test = genfromtxt(
        clf_model_conf["evaluate"]["paths"]["y_test_in_path"],
        delimiter=",",
        skip_header=1,
    )
    y_pred_test = genfromtxt(
        clf_model_conf["evaluate"]["paths"]["y_pred_test_in_path"], delimiter=","
    )

    # Get X_train and y_train
    train_eval = comb_eval(y=y_train, y_pred=y_pred_train)
    test_eval = comb_eval(y=y_test, y_pred=y_pred_test)

    # Build final eval dict
    train_test_eval = {"train_eval": train_eval, "test_eval": test_eval}

    # Save Evaluation
    with open(clf_model_conf["evaluate"]["paths"]["results_out_path"], "w") as f:
        json.dump(train_test_eval, f, indent=4)

    logger.error("Model evaluation finished!")
