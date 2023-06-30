from sales.entity.artifact_entity import RegressionMetricArtifact
from sales.exception import SalesException
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


import os, sys


def get_regression_score(y_true, y_pred) -> RegressionMetricArtifact:
    try:
        model_mean_squared_error = mean_squared_error(y_true, y_pred)
        model_mean_absolute_error = mean_absolute_error(y_true, y_pred)
        model_r2_score = r2_score(y_true, y_pred)

        regression_metric = RegressionMetricArtifact(
            mean_squared_error=model_mean_squared_error,
            mean_absolute_error=model_mean_absolute_error,
            r2_score=model_r2_score,
        )

        return regression_metric

    except Exception as e:
        raise SalesException(e, sys)
