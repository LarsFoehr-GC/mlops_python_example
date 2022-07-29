""" This module contains the specific prediction function for FastAPI.

"""
import glob
from joblib import load
import numpy as np
from typing import Dict


def preprocess_data_api(new_data: np.array) -> np.array:

    PIPE_PATH = glob.glob("**/preprocess_pipe_trained.joblib", recursive=True)[0]
    preprocess_pipe = load(PIPE_PATH)

    new_data = preprocess_pipe.transform(new_data)


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

    x = np.array([preg, gluco, bp, ins, bmi, dpf, age])
    x = preprocess_data_api(x)

    y = clf_model.predict(x)[0]  # just get single value
    prob = clf_model.predict_proba(x)[0].tolist()  # send to list for return

    return {"prediction": int(y), "probability": prob}
