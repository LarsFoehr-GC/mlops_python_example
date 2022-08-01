""" This module contains the specific prediction function for FastAPI.

"""
import glob
from joblib import load
import numpy as np
import pandas as pd
from typing import Dict


def preprocess_data_api(new_data: pd.DataFrame) -> np.array:
    """Preprocess new incoming data to be predicted.

    Args:
        new_data (pd.DataFrame): New data to be predicted.

    Returns:
        new_data: Preprocessed data as np.arrray.

    """

    PIPE_PATH = glob.glob("**/preprocess_pipe_trained.joblib", recursive=True)[0]
    preprocess_pipe = load(PIPE_PATH)

    new_data = preprocess_pipe.transform(new_data)

    return new_data


def get_prediction_api(
    clf_model,
    preg: int,
    gluco: int,
    bp: int,
    ins: int,
    bmi: float,
    dpf: float,
    age: int,
) -> Dict:
    """Get predictions for all input variables

    Args:
        preg (int): Number of pregnancies
        gluco (int): Glucose level
        bp (int): Blood pressure
        ins (int): Insulin Level
        bmi (float): Body mass index
        dpf (float): Diabetes pedigree function
        age (int): Age

    Returns:
        Dictionary containing predictions and probaility predictions.

    """

    # Build DataFrame from new data.
    X = pd.DataFrame(
        {
            "Pregnancies": [preg],
            "Glucose": [gluco],
            "BloodPressure": [bp],
            "Insulin": [ins],
            "BMI": [bmi],
            "DiabetesPedigreeFunction": [dpf],
            "Age": [age],
        }
    )

    X = preprocess_data_api(new_data=X)

    y = clf_model.predict(X)[0]  # just get single value
    prob = clf_model.predict_proba(X)[0].tolist()  # send to list for return

    return {"prediction": int(y), "probability": prob}
