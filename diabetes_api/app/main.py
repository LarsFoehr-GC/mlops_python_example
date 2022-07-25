""" This module builds a small API with FastAPI.

The model is imported and then used to makes predictions for newly delivered data.

"""
from fastapi import FastAPI
from joblib import load
from typing import Dict

from classification_model.predict import get_prediction
from diabetes_api.app import __version__ as app_version
from diabetes_api.app.config import PROJECT_NAME
from diabetes_api.schemas import Health, ModelParams
from models import __version__ as model_version

# Start FastAPI
app = FastAPI()


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
    MODEL_PATH = "/Users/larsmoller/Desktop/Aktuell/MLOps_UseCases/Praxis_Projekte/mlops_python_example/models/model.joblib"
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
