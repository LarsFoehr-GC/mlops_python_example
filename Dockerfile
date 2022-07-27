# Start Python 3.10 Image
FROM python:3.10

# Set temporary working directory
WORKDIR /code

# Copy repo into container --- This is bad style, because not everything is needed!
COPY . /code/

# Install dependencies
RUN pip install --no-cache-dir -r --upgrade /code/requirements.txt

# Run the app via uvicorn
CMD ["uvicorn", "diabetes_api.app.main:app", "--host", "0.0.0.0", "--port", "80"]
