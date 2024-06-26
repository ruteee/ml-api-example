# Contents
This repository contains a training pipeline for the iris-dataset classification and a REST API for model utilization that can be consumed by other services.

### Instruction on how to reproduce the environment for running both the pipeline and the API:
  The repo contains a file denoted 'requirements.txt'. You can create your Python virtualenv and then run the following command:
  - $ pip install -r requirements.txt

# ML model Training pipeline for IRIS
 The pipeline is written in Python and performs the following steps:
 - Missing data imputation
 - Normalization 
 - Model fitting with GridSearchCV

### How to run the training pipeline:
  - The training pipeline is implemented on file modeling/modeling.py
  - For running in a  Python environment, run the following command:
     - $ python modeling/modeling.py 

The model chosen for the application was the LogisticRgressor from sklearn, and the training process performs a cross-validation with 5 folds, searching over a domain of 3 parameters. 

# REST API for model utilization 
For the model utilization, a REST API was implemented. The API is written in Python and uses the Flask framework. The API contains a POST endpoint for performing the prediction under the route: classify/iris

A validation stage was also implemented. Also, a set of unitary tests written with Pytest is available.


### How to init the API:
   - For running in a  Python environment, run the following command:
     - $ python app/api.p 
### How to request to the API:
  - The API accepts only POST requests, in the route: classify/iris for performing the Iris classification. The method receives a JSON in the following       format:
      {
          "sepal_width": 1.0,
          "sepal_length":  1.5,
          "petal_length":  1.0,
          "petal_width":  0.5
      }

  The values for each of the four arguments must be float or int. 

  - The API returns a response JSON containing the classification as shown in the example below:
    {
        "iris-classification": "Iris-setosa" 
    }

  - An example snippet for requesting from a Python script is shown below:
    
      import requests
      data = {
        'sepal_width': 1.0,
        'sepal_length':  0.5,
        'petal_length':  2.0,
        'petal_width':  2.5
      }
      res = requests.post(url="http://127.0.0.1:8080/classify/iris", json=data)
   - It is also possible to request from a GUI client such as Insomnia, Postman, or using curl:
  - e.g: curl -d '{"sepal_width":1.0, "sepal_length":0.5, "petal_length": 2.0, "petal_width": 2.5}' -H "Content-Type: application/json" -X POST http://localhost:8080/classify/iris

# Extras
The repository also contains a Dockerfile for the API, that can be used to deploy the application on a server
 For building the image, run:
  - docker build . -t api:v1
 For running the image:
  - docker run api:v1

  
