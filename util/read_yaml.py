""" This module contains a function to easily read in yaml files.

"""
from typing import Dict
import glob
import yaml


def read_yaml(file_name: str) -> Dict:
    """Search for specific yaml file and read it in.

    Args:
        file_name (str): Name of the yaml file

    Returns:
        yaml_file: Dictionary containing the information from the yaml file.

    """
    # Build yaml path
    yaml_path = glob.glob("**/classification_model_config.yaml", recursive=True)[0]

    # Read yaml file
    with open(yaml_path) as file:
        yaml_file = yaml.safe_load(file)

    return yaml_file
