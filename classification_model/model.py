""" This module contains the building of the model grid search CV.

This GridSearchCV Object will be written out.
The actual training follows in the next step.

"""
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV

from util.logger import define_logger
from util.read_yaml import read_yaml

logger = define_logger()

if __name__ == "__main__":

    logger.error("Build the model grid search cv ...")

    # Get classification model config yaml file
    clf_model_conf = read_yaml("classification_model_config.yaml")

    # Get information from config yaml file
    n_splits = clf_model_conf["model"]["k_fold"]["n_splits"]

    # Model parameters
    params = clf_model_conf["model"]["params"]

    # Instantiate Basic Model
    lr = LogisticRegression()

    # Build K-Fold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=5)

    # Build Gridsearch CV
    lr_cv = GridSearchCV(lr, params, cv=kf)

    # Save the Grid Search as joblib to used in training step
    dump(lr_cv, clf_model_conf["model"]["paths"]["model_out_path"])
