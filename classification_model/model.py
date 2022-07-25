""" This module contains functions around the classification model and the model pipeline.

"""

import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict

# Build logger
logging.basicConfig(
    format="%(asctime)s; %(levelname)s - %(name)s - %(message)s", level=logging.DEBUG
)
logger = logging.getLogger(__name__)


def get_variables(data: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Seperate the target variable from the feature matrix.

    y is the targed variable.
    X is the feature matrix.

    Args:
        data (pd.DataFrame): The dataframe to be worked with.

    Returns:
        X, y: Features matrix and targed variable.

    """

    y = data[column]
    X = data.drop([column], axis=1)

    return X, y


def train(DATA_PATH: str) -> Tuple[Pipeline, dict]:
    """Train the model via ML Pipeline.

    Args:
        DATA_PATH (str): Path to the preprocessed data, to be trained.

    Returns:
        X, y: Features matrix and targed variable.

    """

    logger.info("Training started ...")

    # Load the preprocessed data set
    df_to_train = pd.read_csv(DATA_PATH)

    # Seperate the target variable from the feature matrix
    X, y = get_variables(df_to_train, "Outcome")

    # Split Training and Test Set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    # Build Training Pipeline
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "LR",
                LogisticRegression(random_state=0),
            ),
        ]
    )

    # Fit the pipeline
    training_logs = pipe.fit(X_train, y_train)

    # Save the training logs
    logs = {"training_logs": training_logs}

    logger.info("Training finished!")

    return pipe, logs


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


def evaluate(DATA_PATH: str, pipeline: Pipeline) -> Dict:
    """Evaluate the trained model.

    Args:
        DATA_PATH (str): Path to the trained data.
        pipeline (Pipeline): Pipeline that was used for training.

    Returns:
        logs: Dictionary containing several evaluation KPIs.

    """

    logger.info("Training started ...")

    # Load the preprocessed data set
    df_to_eval = pd.read_csv(DATA_PATH)

    # Seperate the target variable from the feature matrix
    X, y = get_variables(df_to_eval, "Outcome")

    # Split data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    # Make predictions and build dictionary containing several evaluation KPIs
    y_pred_test = pipeline.predict(X_test)
    test_result = comb_eval(y_test, y_pred_test)

    # Calculate true and false prositive rates
    dummy_probs = [0 for _ in range(len(y_test))]
    model_probs = pipeline.predict_proba(X_test)
    model_probs = model_probs[:, 1]
    dummy_fpr, dummy_tpr, _ = roc_curve(y_test, dummy_probs)
    model_fpr, model_tpr, _ = roc_curve(y_test, model_probs)

    # Build precision recall values
    y_scores = pipeline.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)

    logs = {
        "metrics": test_result,
        "roc_curve": {
            "model_tpr": model_tpr,
            "model_fpr": model_fpr,
            "dummy_tpr": dummy_tpr,
            "dummy_fpr": dummy_fpr,
        },
        "precision_recall_curve": {
            "precisions": precisions,
            "recalls": recalls,
            "thresholds": thresholds,
        },
    }

    return logs
