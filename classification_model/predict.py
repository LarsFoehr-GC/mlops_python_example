""" This modul contains the function(s) to predict the output of new data.

"""
from joblib import load
from numpy import genfromtxt, savetxt
import pandas as pd

from util.logger import define_logger
from util.read_yaml import read_yaml

logger = define_logger()

if __name__ == "__main__":

    logger.error("Model prediction started ...")

    # Get classification model config yaml file
    clf_model_conf = read_yaml("classification_model_config.yaml")

    # Load the GridSearchCV
    lr = load(clf_model_conf["predict"]["paths"]["model_in_path"])

    # Get X_train and X_test
    X_train = genfromtxt(
        clf_model_conf["predict"]["paths"]["X_train_in_path"], delimiter=","
    )
    X_test = genfromtxt(
        clf_model_conf["predict"]["paths"]["X_test_in_path"], delimiter=","
    )

    # Make predictions
    y_pred_train = lr.predict(X_train)
    y_pred_test = lr.predict(X_test)

    # Save y_pred and y_pred_proba
    savetxt(
        clf_model_conf["predict"]["paths"]["y_pred_train_out_path"],
        y_pred_train,
        delimiter=",",
    )
    savetxt(
        clf_model_conf["predict"]["paths"]["y_pred_test_out_path"],
        y_pred_test,
        delimiter=",",
    )

    logger.error("Model prediction finished!")
