""" This module contains the parameters for the Health feedback side of the API.

"""
from pydantic import BaseModel


class Version(BaseModel):
    name: str
    api_version: str
    model_version: str
