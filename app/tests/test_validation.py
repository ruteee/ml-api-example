import pytest
from app.api import application
import json


def setup():
    application.app_context().push()
    
@pytest.fixture()
def client():
    setup()
    return application.test_client()


def test_negative_number_validation(client):
    response = client.post("/classify/iris", content_type='application/json', data = json.dumps({
        "sepal_width": 1,
        "sepal_length":  -8,
        "petal_length":  1,
        "petal_width":  0.5
        })
    )

    assert response.text == 'sepal_length accept positive values'
    assert response.status_code == 400

def test_incorrect_param_and_negative_validation(client):
    response = client.post("/classify/iris", content_type='application/json', data = json.dumps({
        "sepal_wid": 1,
        "sepal_length":  -8,
        "petal_length":  1,
        "petal_width":  0.5
        })
    )

    assert response.text == "['sepal_length accept positive values', 'sepal_width is required']"
    assert response.status_code == 400

def test_incorrect_param_validation(client):
    response = client.post("/classify/iris", content_type='application/json', data = json.dumps({
        "sepal_wid": 1,
        "sepal_length":  0,
        "petal_length":  1,
        "petal_width":  0.5
        })
    )

    assert response.text == "['sepal_width is required']"
    assert response.status_code == 400


def test_overflow_validation(client):
    response = client.post("/classify/iris", content_type='application/json', data = json.dumps({
        "sepal_width": 2**1980,
        "sepal_length":  0,
        "petal_length":  1,
        "petal_width":  0.5
        })
    )

    assert response.text == "Range value not acceptable"
    assert response.status_code == 400


def test_incorrect_type_validation(client):
    response = client.post("/classify/iris", content_type='application/json', data = json.dumps({
        "sepal_width": 1,
        "sepal_length":  "0",
        "petal_length":  1,
        "petal_width":  0.5
        })
    )

    assert response.text == "['Field sepal_length accepts float or int type']"
    assert response.status_code == 400


def test_incorrect_param_and_type_validation(client):
    response = client.post("/classify/iris", content_type='application/json', data = json.dumps({
        "sepal_wid": 1,
        "sepal_length":  "0",
        "petal_length":  1,
        "petal_width":  0.5
        })
    )
    assert response.text == "['Field sepal_length accepts float or int type', 'sepal_width is required']"
    assert response.status_code == 400
