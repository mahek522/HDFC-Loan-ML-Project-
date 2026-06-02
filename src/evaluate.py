import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

def evaluate_classification(
        y_true,
        y_pred):
    return {
        "Accuracy":
        accuracy_score(y_true, y_pred),
        "Precision":
        precision_score(y_true, y_pred),
        "Recall":
        recall_score(y_true, y_pred),
        "F1":
        f1_score(y_true, y_pred)
    }


def evaluate_regression(
        y_true,
        y_pred):
    mae = mean_absolute_error(
        y_true,
        y_pred
    )
    mse = mean_squared_error(
        y_true,
        y_pred
    )
    rmse = np.sqrt(mse)
    r2 = r2_score(
        y_true,
        y_pred
    )

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }