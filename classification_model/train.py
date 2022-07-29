""" This module trains the grid search CV built in the former step.

At the end the trained model will be saved to be used for prediction and evaluation.

"""
from joblib import dump, load
from numpy import genfromtxt

from util.logger import define_logger
from util.read_yaml import read_yaml

logger = define_logger()

if __name__ == "__main__":

    logger.error("Model training started ...")

    # Get classification model config yaml file
    clf_model_conf = read_yaml("classification_model_config.yaml")

    # Load the GridSearchCV
    lr_cv = load(clf_model_conf["train"]["paths"]["model_in_path"])

    # Get X_train and y_train
    X_train = genfromtxt(
        clf_model_conf["train"]["paths"]["X_train_in_path"], delimiter=","
    )
    y_train = genfromtxt(
        clf_model_conf["train"]["paths"]["y_train_in_path"],
        delimiter=",",
        skip_header=1,
    )

    # Fit the GridSearchCV
    lr_cv.fit(X_train, y_train)

    # Get the best possible model
    lr_trained = lr_cv.best_estimator_

    # Save the Grid Search as joblib to used in training step
    dump(lr_trained, clf_model_conf["train"]["paths"]["model_out_path"])

    logger.error("Model training finished!")
