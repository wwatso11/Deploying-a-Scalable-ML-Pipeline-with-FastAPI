import pytest
import numpy as np
import os
import pandas as pd

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
    # add description for the first test
    """
    project_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(project_path, "model", "model.pkl")
    model = load_model(model_path) 

    df = pd.DataFrame(np.random.rand(10, 108))
    preds = inference(model, df)

    assert all(x in [0, 1] for x in preds)



# TODO: implement the second test. Change the function name and input as needed
def test_two():
    """
    # add description for the second test
    """
    # Your code here
    pass


# TODO: implement the third test. Change the function name and input as needed
def test_three():
    """
    # add description for the third test
    """
    # Your code here
    pass
