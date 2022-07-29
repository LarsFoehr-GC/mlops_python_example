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

## Build of the pipeline
The following commands were used to build the pipeline:

#### Process raw daata
```
dvc run -n process_raw -d classification_model/process_raw_data.py -d data/diabetes_raw.csv -o data/diabetes_raw_processed.csv python3 classification_model/process_raw_data.py
```
#### Split data preprocess
```
dvc run -n split_preprocess -d classification_model/split_preprocess.py -d data/diabetes_raw_processed.csv -o data/X_train.csv -o data/X_test.csv -o data/y_train.csv -o data/y_test.csv python3 classification_model/split_preprocess.py
```
#### Build the model
```
dvc run -n model -d classification_model/split_preprocess.py -d classification_model/model.py -o models/grid_search_cv.joblib python3 classification_model/model.py
```
#### Train the model
```
dvc run -n train -d classification_model/model.py -d classification_model/train.py -d models/grid_search_cv.joblib -d data/X_train.csv -d data/y_train.csv -O models/model_trained.joblib python3 classification_model/train.py
```
#### Predict
```
dvc run -n predict -d classification_model/train.py -d classification_model/predict.py -d models/model_trained.joblib -d data/X_train.csv -d data/X_test.csv -o data/y_pred_train.csv -o data/y_pred_test.csv python3 classification_model/predict.py
```
#### Evaluation
```
dvc run -n evaluate -d classification_model/predict.py -d classification_model/evaluate.py -d data/y_train.csv -d data/y_pred_train.csv -d data/y_test.csv -d data/y_pred_test.csv -M results/metrics.json python3 classification_model/evaluate.py
```
## Experiment
- dvc pull
- dvc run ...
- dvc push

# Fast API
- https://fastapi.tiangolo.com/#run-it
- https://fastapi.tiangolo.com/deployment/docker/

### Call

```
uvicorn diabetes_api.app.main:app --reload
```

# Docker
Fast API is called via Docker. The specification of the Docker Image can be found in the Dockerfile.

## Download
Before you can actually use Docker, you have to download it: [Link](https://www.docker.com/products/docker-desktop/)

## Examples to build and run and the image on localhost
docker build -t mlops_py_image .
docker run -d --name mlops_py_container -p 80:80 mlops_py_image

# Azure Deployment
https://towardsdatascience.com/deploy-fastapi-on-azure-with-github-actions-32c5ab248ce3

# To Do's:
- pre-commit with mypy but only for important modules
- Explain DVC in README
- Explain whole workflow
- Write better tests
- Think about Model monitoring
- Think about automatic retraining the model
- Refactoring: Write better modules and functions!
- Better work with paths!
