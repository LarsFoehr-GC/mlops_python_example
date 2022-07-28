# Start Python 3.10 Image
FROM python:3.10

# Set temporary working directory
WORKDIR /code

# Copy repo into container --- This is bad style, because not everything is needed!
COPY ./requirements.txt /code/
COPY ./diabetes_api /code//diabetes_api/
COPY ./classification_model/ /code/classification_model/
COPY ./models/ /code//models
COPY ./dvc.lock /code/
COPY ./dvc.yaml /code/

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Run the app via uvicorn
CMD ["uvicorn", "diabetes_api.app.main:app", "--host", "0.0.0.0", "--port", "80"]
