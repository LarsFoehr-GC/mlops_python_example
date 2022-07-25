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
