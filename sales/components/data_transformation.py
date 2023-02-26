import sys

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sales.constant.training_pipeline import TARGET_COLUMN
from sales.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact,
)

from sales.entity.config_entity import DataTransformationConfig
from sales.logger import logging
from sensor.utils.main_utils import save_numpy_array_data, save_object
from sales.exception import SalesException

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


class DataTransformation:
    def __init__(
        self,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig,
    ):
        """
        :param data_validation_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: configuration for data transformation
        """
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config

        except Exception as e:
            raise SalesException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise SalesException(e, sys)
    

    def get_data_transformer_object(self)->ColumnTransformer:
        try:
            schema_file_path = self.data_validation_artifact.schema_file_path
            
            dataset_schema = read_yaml_file(file_path=schema_file_path)
            
            numerical_columns = dataset_schema[NUMERICAL_COLUMN_KEY]
            categorical_columns = dataset_schema[CATEGORICAL_COLUMN_KEY]
            
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy="median")),
                ('scaler', StandardScaler())
            ]
            )

            cat_pipeline = Pipeline(steps=[
                 ('impute', SimpleImputer(strategy="most_frequent")),
                 ('one_hot_encoder', OneHotEncoder(sparse=False,handle_unknown='ignore')),
                 ('scaler', StandardScaler(with_mean=False))
            ]
            )
            
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
            
            preprocessing = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns),
            ])
            return preprocessing
            
        except Exception as e:
            raise SalesException(e,sys)
        

