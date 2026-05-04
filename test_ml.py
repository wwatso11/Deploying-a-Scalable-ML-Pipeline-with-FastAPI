import pytest
import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# TODO: add necessary import

# TODO: implement the first test. Change the function name and input as needed
def test_inference():
    """
    Test that inference gives a value between 0 and 1
    """
    project_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(project_path, "model", "model.pkl")
    model = load_model(model_path) 

    df = pd.DataFrame(np.random.rand(10, 108))
    preds = inference(model, df)

    assert all(x in [0, 1] for x in preds)



# TODO: implement the second test. Change the function name and input as needed
def test_compute_model_metrics():
    """
    Test that compute_model_metrics returns precision, recall, and fbeta
    values within the expected range [0, 1].
    """
    y_true = np.array([1, 0, 1, 1, 0])
    preds = np.array([1, 0, 0, 1, 1])

    precision, recall, fbeta = compute_model_metrics(y_true, preds)

    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1


# TODO: implement the third test. Change the function name and input as needed
def test_train_model():
    """
    Test that train_model returns a RandomForestClassifier
    and that the model is fitted.
    """
    X = np.random.rand(20, 5)
    y = np.random.randint(0, 2, 20)

    model = train_model(X, y)

    # Check model type
    assert isinstance(model, RandomForestClassifier)

    # Check model is fitted (has learned attributes)
    assert hasattr(model, "estimators_")