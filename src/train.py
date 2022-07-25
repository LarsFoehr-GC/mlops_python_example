""" This module trains the data with model, that was defined in model.py.

At the end the trained model is dumped as a joblib file.

"""

import importlib.util
import joblib
import logging
import os
import sys

# Build logger
logging.basicConfig(
    format="%(asctime)s; %(levelname)s - %(name)s - %(message)s", level=logging.DEBUG
)
logger = logging.getLogger(__name__)


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


# Set paths (TODO: Write a good config file)
DATA_PATH = os.path.abspath(sys.argv[1])
MODEL_PATH = sys.argv[2]

sys.path.insert(1, MODEL_PATH)

model = module_from_file("model", MODEL_PATH)

if __name__ == "__main__":
    logger.info("Training started ...")
    pipeline, log_train = model.train(DATA_PATH)
    joblib.dump(pipeline, "./models/model.joblib")
    logger.info("Training finished!")
