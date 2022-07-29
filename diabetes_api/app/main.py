""" This module builds a small API with FastAPI.

The model is imported and then used to makes predictions for newly delivered data.

"""
from fastapi import FastAPI
import glob
from joblib import load
import json
import os
from typing import Dict

from diabetes_api.app.predict_api import get_prediction_api
from diabetes_api.app import __version__ as app_version
from diabetes_api.app.config import PROJECT_NAME
from diabetes_api.schemas import Version, ModelParams
from models import __version__ as model_version

from util.logger import define_logger

logger = define_logger()

# Feedback for current working directory
cwd = os.getcwd()
logger.error("Aktueller Pfad: %s", cwd)

# Start FastAPI
description = """
Diabetes Prediction API helps to predict Diabetes. 🚀

## Sites

Right now there are three pages.

* **Root** : Gives feedback whether or not the API works.
* **/version** : Shows version of the API and the model.
* **/metrics : Shows current training and test evaluation metrics.
* **/predict** : Prediction site.
"""

app = FastAPI(
    title="Diabetes Prediction API", description=description, version=app_version
)


@app.get("/")
def read_root():
    return {"Yes! ": "This API works!"}


@app.get("/version", response_model=Version, status_code=200)
def version() -> Dict:
    """
    This page shows the current version of the API and the model.
    """
    version = Version(
        name=PROJECT_NAME, api_version=app_version, model_version=model_version
    )

    return version.dict()


@app.get("/metrics", status_code=200)
def metrics() -> Dict:
    """
    This page shows current training and test evaluation metrics.

    """
    # Read in JSON File
    METRICS_PATH = glob.glob("**/metrics.json", recursive=True)[0]

    with open(METRICS_PATH, "r") as f:
        data = json.load(f)

    return data


@app.post("/predict")
def predict(params: ModelParams) -> Dict:
    """
    On this page predictions can be made.

    """

    # Load the model
    logger.error("Modell Pfad: %s", glob.glob("**/model.joblib", recursive=True))

    MODEL_PATH = glob.glob("**/model_trained.joblib", recursive=True)[0]

    logger.error("Modell Pfad: %s", MODEL_PATH)

    classifier_model = load(MODEL_PATH)

    pred = get_prediction_api(
        clf_model=classifier_model,
        preg=params.preg,
        gluco=params.gluco,
        bp=params.bp,
        ins=params.ins,
        bmi=params.bmi,
        dpf=params.dpf,
        age=params.age,
    )

    return pred
