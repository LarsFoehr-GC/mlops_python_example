# mlops_python_example
Example MLOps workflow with Python.

# Overview
This repo contains three main parts.
- Model Building
- API, Docker, Azure
- CI/CD

## Model Building
On the model side DVC is used to build an ML Workflow. Each step in the model building can be rebuilt. It is also possible to further develop the model.

## API in docker container
The API over which the model can be called is built with FastAPI. The API is inside a docker container, which is hosted via Microsoft Azure.

## CI/CD
Github is used as Code repository as well as for CI/CD. With each release, the new API (and model) redeployed on Microsoft Azure. Furthermore with each Push into main Unit tests are run.

# Configuration

## First steps, to get running
**Build venv**
```
source mlops_py_env/bin/activate
```

**Get dependencies**
```
python -m pip install -r requirements.txt
```

**Install pre-commit**
```
pre-commit install
```

**Set PYTHONPATH inside venv**
```
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## New packages
**Install new packages**
```
install packages with pip install PACKAGE
```

**Add new package to requirement**
```
python -m pip freeze > requirements.txt
```

# DVC
DVC is used to build ML workflow and to track data (and the model in future).

## Get connection to Azure Blob Storage
The training data is stored inside my own data blob storage. To build your own data blob storage and store the data there, follow the instructions on this [link](https://dvc.org/blog/azure-remotes-in-dvc)

## Build of the pipeline
The following commands were used to build the pipeline:

**Process raw data**
```
dvc run -n process_raw -d classification_model/process_raw_data.py -d data/diabetes_raw.csv -o data/diabetes_raw_processed.csv python3 classification_model/process_raw_data.py
```
**Split data preprocess**
```
dvc run -n split_preprocess -d classification_model/split_preprocess.py -d data/diabetes_raw_processed.csv -o data/X_train.csv -o data/X_test.csv -o data/y_train.csv -o data/y_test.csv python3 classification_model/split_preprocess.py
```
**Build the model**
```
dvc run -n model -d classification_model/split_preprocess.py -d classification_model/model.py -o models/grid_search_cv.joblib python3 classification_model/model.py
```
**Train the model**
```
dvc run -n train -d classification_model/model.py -d classification_model/train.py -d models/grid_search_cv.joblib -d data/X_train.csv -d data/y_train.csv -O models/model_trained.joblib python3 classification_model/train.py
```
**Predict**
```
dvc run -n predict -d classification_model/train.py -d classification_model/predict.py -d models/model_trained.joblib -d data/X_train.csv -d data/X_test.csv -o data/y_pred_train.csv -o data/y_pred_test.csv python3 classification_model/predict.py
```
**Evaluation**
```
dvc run -n evaluate -d classification_model/predict.py -d classification_model/evaluate.py -d data/y_train.csv -d data/y_pred_train.csv -d data/y_test.csv -d data/y_pred_test.csv -M results/metrics.json python3 classification_model/evaluate.py
```
## Experiment with new model
You can further develop the model and experiment with it. The woflow would be:

- Get the training data (and in future the model): dvc pull
- Develop the model
- Check the model metrics locally
- Rerun the ML Workflow: dvc repro
- dvc add
- dvc push
- git add .
- git commit -m "Your message"
- git push

## Further information
Further information how to use DVC can be read in this [tutorial](https://mlops-guide.github.io/Versionamento/)

# Fast API
The model can be called via FastAPI. How you can build and run an API via FastAPI, you can see learn [here](https://fastapi.tiangolo.com/#example)

## Call
You can call the API locally with this command:
```
uvicorn diabetes_api.app.main:app --reload
```

# Docker
Fast API is run inside a docker container. The specification of the Docker Image can be found in the Dockerfile.

## Download
Before you can actually use Docker, you have to download it: [Link](https://www.docker.com/products/docker-desktop/)

## Build and run docker image locally
**Build**
```
docker build -t mlops_py_image .
```
**Run**
```
docker run -d --name mlops_py_container -p 80:80 mlops_py_image
```

## Deeper information
Further information on FastAPI deployment via docker can be found [here](https://fastapi.tiangolo.com/deployment/docker/)


# Azure Deployment
The API is deployed on Microsoft Azure. Therefore the docker image is stored in Azure Container Regristry and than loaded into a App Service to get the API running.

With each release, this process runs automatically via Github Actions. How this can be done, you can find [here](https://towardsdatascience.com/deploy-fastapi-on-azure-with-github-actions-32c5ab248ce3).

## Link
The current API can be used with this [link](https://mlops-py-fastapi.azurewebsites.net/docs)

# Github
Besides the automatic deployment of the API, there is also a Github actions which runs unit tests with each Pull Request into main.

# To Do's:

## Refactoring
Refactoring: Write better modules, function, test and paths.

## Github
Develop the repo. Protect the main branch and adjust the overall setting.

## Tests
Write better tests. Right now, there is just one dummy test to check if the CI works on Github.

## Pre-Commit
Develop [pre-commit](https://pre-commit.com/#intro). Especially connect it with mypy so that Static Type checking is run with every commit -> [Link](https://github.com/pre-commit/mirrors-mypy).

## Model Monitoring
Right now the MLOPs Workflow stops with done Model and the API. Think about Model Monitoring and trigger points, when the model has to be retrained.

Here are three Tutorial which can be used for orientation:
- [Monitoring Machine Learning Models in Production](https://www.google.de/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjxt92z4Kf5AhWPlqQKHedoD0cQFnoECBIQAQ&url=https%3A%2F%2Ftowardsdatascience.com%2Fmonitoring-machine-learning-models-in-production-how-to-track-data-quality-and-integrity-391435c8a299&usg=AOvVaw1R0bBDcc19aTTXTTetDPIq)
- [Essential guide to Machine Learning Model Monitoring in Production](https://towardsdatascience.com/essential-guide-to-machine-learning-model-monitoring-in-production-2fbb36985108)
- [Monitor Your Model Performance with Python Streamlit](https://medium.com/dataman-in-ai/monitor-your-model-performance-with-python-streamlit-e0db20f09023)
- [A Machine Learning Model Monitoring Checklist: 7 Things to Track](https://www.kdnuggets.com/2021/03/machine-learning-model-monitoring-checklist.html)

## Automatic Model Retraining
As mentioned above we need a trigger point, at which to retrain the model. This could be done via model monitoring and then start DVC again via dvc repro.
