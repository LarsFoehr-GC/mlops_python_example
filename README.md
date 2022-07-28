# mlops_python_example
Example MLOps workflow with Python.

## Ongoing Documentation

### 1) Configuration
build venv: source mlops_py_env/bin/activate

get dependencies: python -m pip install -r requirements.txt

install packages with pip install PACKAGE

Add package to dependencies: python -m pip freeze > requirements.txt

To install pre-commit: pre-commit install

To set PYTHONPATH inside venv: export PYTHONPATH=$PYTHONPATH:$(pwd)

# Work with dvc

## Get connection

## Experiment
- dvc pull
- dvc run ...
- dvc push

## To Do's:
- pre-commit with mypy but only for important modules
- Explain DVC in README
- Explain whole workflow
- Write better tests
- Think about Model monitoring
- Think about automatic retraining the model
- Refactoring: Write better modules and functions!
- Better work with paths!

# Fast API
https://fastapi.tiangolo.com/#run-it

https://fastapi.tiangolo.com/deployment/docker/

# Docker
Fast API is called via Docker. The specification of the Docker Image can be found in the Dockerfile.

## Download
Before you can actually use Docker, you have to download it: [Link](https://www.docker.com/products/docker-desktop/)

## Examples to build and run and the image on localhost
docker build -t mlops_py_image .
docker run -d --name mlops_py_container -p 80:80 mlops_py_image

# Azure Deployment
https://towardsdatascience.com/deploy-fastapi-on-azure-with-github-actions-32c5ab248ce3
