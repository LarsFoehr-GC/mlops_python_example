""" This module builds a small API with FastAPI.

The model is imported and then used to makes predictions for newly delivered data.

"""
from fastapi import FastAPI
import glob
from joblib import load
import logging
import os
from typing import Dict

from diabetes_api.app.predict_api import get_prediction
from diabetes_api.app import __version__ as app_version
from diabetes_api.app.config import PROJECT_NAME
from diabetes_api.schemas import Health, ModelParams
from models import __version__ as model_version

# Build logger
logging.basicConfig(
    format="%(asctime)s; %(levelname)s - %(name)s - %(message)s", level=logging.ERROR
)
logger = logging.getLogger(__name__)
logger.error("App läuft")

cwd = os.getcwd()
logger.error("Aktueller Pfad: %s", cwd)

# Start FastAPI

description = """
Diabetes Prediction API helps to predict Diabetes. 🚀

## Sites

Right now there are three sites.

* **Root** : Gives feedback whether or not the API works.
* **/health** : Shows information about the API.
* **/predict** : Prediction site.
"""

app = FastAPI(
    title="Diabetes Prediction API", description=description, version=app_version
)


@app.get("/")
def read_root():
    return {"Yes! ": "This API works!"}


@app.get("/health", response_model=Health, status_code=200)
def health() -> Dict:
    """
    Root Get
    """
    health = Health(
        name=PROJECT_NAME, api_version=app_version, model_version=model_version
    )

    return health.dict()


@app.post("/predict")
def predict(params: ModelParams) -> Dict:

    # Load the model

    logger.error("Modell Pfad: %s", glob.glob("**/model.joblib", recursive=True))

    MODEL_PATH = glob.glob("**/model.joblib", recursive=True)[0]

    logger.error("Modell Pfad: %s", MODEL_PATH)

    classifier_model = load(MODEL_PATH)

    pred = get_prediction(
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
