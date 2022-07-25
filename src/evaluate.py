""" This module evaluates the trained model
"""
import importlib.util
import json
import joblib
import logging
import os
import sys

# Build logger
logging.basicConfig(
    format="%(asctime)s; %(levelname)s - %(name)s - %(message)s", level=logging.DEBUG
)
logger = logging.getLogger(__name__)

# Set variables
DATA_PATH = os.path.abspath(sys.argv[1])
MODEL_PATH = sys.argv[2]
JOBLIB_PATH = sys.argv[3]

sys.path.insert(1, MODEL_PATH)


def module_from_file(module_name, file_path):
    """This function gets a module from a specific file path.

    Args:
        module_name (str): Name of the module to be loaded.
        file_path (str): Filepath which leads to the module.

    Returns:
        module: The module to be used.

    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


with open("./results/metrics.json", "w") as outfile:
    logger.info("Evaluation started ...")
    model = module_from_file("model", MODEL_PATH)
    pipeline = joblib.load(JOBLIB_PATH)
    log_eval = model.evaluate(DATA_PATH, pipeline)
    json.dump(log_eval["metrics"], outfile)
    logger.info("Evaluation finished!")
