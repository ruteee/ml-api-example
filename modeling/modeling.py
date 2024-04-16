import logging
import pickle
import sys

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

logging.basicConfig(level="INFO", stream=sys.stdout)

model_path = "app/assets/iris_classifier.pickle"
def save_model_pickle(model, filename):
    """
    Saves a ML model in pickle format
    Args: model - The model bytes object to be saved
          filename - The filename
    """
    with open(filename, 'wb') as pickle_file:
        pickle.dump(model, pickle_file)


def get_iris_data():
    """
    Get iris dataset from UCI reposiory
    Returns: 
        X  - Dataframe containing 4 features regrding iris characteristics
        y - The target array for the iris classification 
    """
    data_iris = fetch_ucirepo(id=53) 
    X = data_iris.data.features 
    X.rename(columns = {
        'sepal length' : 'sepal_length',
        'sepal width' : 'sepal_width',
        'petal length' : 'petal_length',
        'petal width': 'petal_width'
    }, inplace=True)
    y = data_iris.data.targets['class']
    return X, y


def get_training_pipeline():
    "Defines training pipeline steps and returns the pipeline"
    pipeline = Pipeline(steps = [
        ('Imputer', SimpleImputer(strategy='mean', keep_empty_features=True)),
        ('normalization', StandardScaler()),
        ('estimator', LogisticRegression() )
    ]
    )
    return pipeline

def fit_model(X_train, y_train):
    """
    Fit a ML pipeline using GridSearchCV with 5 folds
    Args: X_train - Dataset to be trained
          y_train - Target column 
    Returns: model - ML pipeline fitted
    """
    parameters = {
        'estimator__solver': ['newton-cg'],
        'estimator__tol': [ 0.0001, 0.003, 0.01],
        'estimator__penalty': [None, 'l2'],
    }

    pipeline = get_training_pipeline()
    model = GridSearchCV(estimator=pipeline,
                            param_grid=parameters,
                            scoring= {"AUC": "roc_auc_ovr"},
                            refit="AUC",
                            cv=5,
                            verbose=1,
                            error_score='raise')
    model = model.fit(X_train, y_train)
    return model


def run_training_pipeline():
    """
        Runs the trainin pipelie for building a classifier for iris dataset and
        saves the trained model
    """

    logging.info(f"Getting dataset")
    X, y = get_iris_data()

    logging.info(f"Spliting dataset")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=14)

    logging.info(f"Fittig model with train data")
    cv_model = fit_model(X_train, y_train)
    y_pred = cv_model.predict(X_test)

    logging.info(f"Computing scores")
    model_score = cv_model.score(X_test, y_test)
    logging.info(f"Model AUC Score: {model_score}")

    test_acc_score = accuracy_score(y_test, y_pred)
    logging.info(f"Accuracy test score: {test_acc_score}")

    logging.info("Saving model")
    save_model_pickle(cv_model, model_path)

if __name__ == '__main__':
    run_training_pipeline()
    