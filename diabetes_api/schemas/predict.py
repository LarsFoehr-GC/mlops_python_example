""" This module contains the parameters for the Predict side of the API.

"""
from pydantic import BaseModel
from typing import Any, List, Optional

# Model parameters
class ModelParams(BaseModel):
    """This class contains all parameters for the model."""

    preg: int = 2
    gluco: int = 183
    bp: int = 74
    ins: int = 88
    bmi: float = 35.3
    dpf: float = 0.158
    age: int = 52
