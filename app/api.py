import sys
import logging
import pickle
import json
from flask import Flask, request, Response
import functools
import pandas as pd

logger = logging.getLogger("a3_api")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

application = Flask(__name__)
model_path = "app/assets/iris_classifier.pickle"


def load_model_pkl(filepath):
    with open(filepath, 'rb') as model_pkl:
        model = pickle.load(model_pkl)
        return model


def validate_params(arg):
    def decorator_validate(func=None):
        @functools.wraps(func)
        def wrapper_validate(*args, **kwargs):
            params = request.json
            valid_data = {}
            expected_keys = ['sepal_length','sepal_width','petal_length','petal_width']
            validation_errors = []
            try:
                for key in expected_keys:
                    if key not in list(params.keys()):
                        error = f"{key} is required"
                        validation_errors.append(error)
                    else: 
                        value = params[key]
                        if type(value) != int and type(value) != float:
                            error = f"Field {key} accepts float or int type"
                            validation_errors.append(error)
                        elif value < 0:
                            error = f"{key} accept positive values"
                            validation_errors.append(error)
                        else:
                            value = float(value)
                            valid_data[key] = [value]
                if validation_errors:
                    logger.error(str(validation_errors))
                    return Response(content_type='application/json', status = 400, response = str(validation_errors))
            except OverflowError:
                error = "Range value not acceptable"
                logger.error(error)
                return Response(content_type='application/json', status = 400, response = str(error))       
            except BaseException as error:
                err = "An internal error occured"
                logger.error(err)
                return Response(content_type='application/json', status = 500, response = str(err))
            return func(params=valid_data)
        return wrapper_validate
    return decorator_validate


@application.route('/classify/iris', methods=['POST'])
@validate_params(None)
def classify_iris(params):
    try:
        logger.info("Receving parammeters")
        dataset = pd.DataFrame.from_dict(params)
        dataset = dataset[['sepal_length','sepal_width','petal_length','petal_width']]

        logger.info("Loading model")
        model = load_model_pkl(model_path)
        iris_classification = model.predict(dataset)            
        model_response = {
            "iris-classification": iris_classification[0]
        }
        response = Response(content_type='application/json', status = 200, response = json.dumps(model_response))
    except FileNotFoundError as err:
        error = "classifier unavaible"
        logger.error(str(err))
        response = Response(content_type='application/json', status = 500, response = json.dumps(error))
    except BaseException as err:
        logger.error(str(err))
        error = "An internal error occurred."
        response = Response(content_type='application/json', status = 500, response = json.dumps(error))
    return response

    
if __name__ == '__main__':
     application.run(host='0.0.0.0', port=8080)
