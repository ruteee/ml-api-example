import pytest
from app.api import application
import json


def setup():
    application.app_context().push()
    
@pytest.fixture()
def client():
    setup()
    return application.test_client()

def test_output_format(client):
    response = client.post("/classify/iris", content_type='application/json', data = json.dumps({
        "sepal_width": 1,
        "sepal_length":  0.5,
        "petal_length":  1,
        "petal_width":  0.5
        })
    )

    data =dict(json.loads(response.data))
    assert list(data.keys()) == ["iris-classification"]
    assert response.status_code == 200
    assert data['iris-classification'] in ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']



