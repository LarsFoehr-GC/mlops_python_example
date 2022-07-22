""" This module trains the data with model, that was defined in model.py.

At the end the trained model is dumped as a joblib file.

"""

import importlib.util
import joblib
import os
import sys


def module_from_file(module_name, file_path):
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
    pipeline, log_train = model.train(DATA_PATH)
    joblib.dump(pipeline, "./models/model.joblib")
