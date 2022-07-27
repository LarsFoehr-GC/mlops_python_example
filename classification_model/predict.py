""" This modul contains the function(s) to predict the output of new data.

"""
from fastapi.logger import logger
import logging

gunicorn_logger = logging.getLogger("gunicorn.error")
logger.handlers = gunicorn_logger.handlers
logger.setLevel(gunicorn_logger.level)

from sklearn.pipeline import Pipeline
from typing import Dict

# Get predictions
def get_prediction(
    clf_model: Pipeline,
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
    logger.info("Prediction beginnt ...")

    x = [[preg, gluco, bp, ins, bmi, dpf, age]]

    y = clf_model.predict(x)[0]  # just get single value
    prob = clf_model.predict_proba(x)[0].tolist()  # send to list for return

    return {"prediction": int(y), "probability": prob}
