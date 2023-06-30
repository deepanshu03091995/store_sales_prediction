import sys
import shutil
import numpy as np
import pandas as pd

from sales.constant.training_pipeline import TARGET_COLUMN
from sales.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact,
)
from sales.entity.config_entity import DataTransformationConfig
from sales.logger import logging
from sales.utils.main_utils import save_numpy_array_data, save_object, read_yaml_file
from sales.exception import SalesException
from sales.constant.training_pipeline import SCHEMA_FILE_PATH
from sklearn.compose import ColumnTransformer

from sales.constant.training_pipeline import PREPROCESSOR_OBJECT_DIR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder


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
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

        except Exception as e:
            raise SalesException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise SalesException(e, sys)

    def get_data_transformer_object(self) -> ColumnTransformer:
        try:
            file_path = self.data_validation_artifact.valid_train_file_path

            numerical_columns = list(self._schema_config["numerical_columns"])
            categorical_columns = list(self._schema_config["categorical_columns"])

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    (
                        "ordinal_encoder",
                        OrdinalEncoder(),
                    ),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )
            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            preprocessing = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessing

        except Exception as e:
            raise SalesException(e, sys)

    def initiate_data_transformation(
        self,
    ) -> DataTransformationArtifact:
        try:

            train_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_train_file_path
            )
            test_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_test_file_path
            )
            preprocessor_object = self.get_data_transformer_object()
            logging.info("Read train and test data completed")

            save_object(PREPROCESSOR_OBJECT_DIR, preprocessor_object)

            logging.info("Obtaining preprocessing object")

            # training dataframe
            input_feature_train_df = train_df.drop(
                list(self._schema_config["drop_columns"]), axis=1
            )
            input_feature_train_df.replace(
                {
                    "Item_Fat_Content": {
                        "low fat": "Low Fat",
                        "LF": "Low Fat",
                        "reg": "Regular",
                    }
                },
                inplace=True,
            )
            target_feature_train_df = train_df[TARGET_COLUMN]

            # testing dataframe
            input_feature_test_df = test_df.drop(
                list(self._schema_config["drop_columns"]), axis=1
            )
            input_feature_test_df.replace(
                {
                    "Item_Fat_Content": {
                        "low fat": "Low Fat",
                        "LF": "Low Fat",
                        "reg": "Regular",
                    }
                },
                inplace=True,
            )
            target_feature_test_df = test_df[TARGET_COLUMN]

            input_feature_train_arr = preprocessor_object.fit_transform(
                input_feature_train_df
            )
            input_feature_test_arr = preprocessor_object.transform(
                input_feature_test_df
            )

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # save numpy array data
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                array=train_arr,
            )
            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                array=test_arr,
            )
            save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor_object,
            )

            """preparing artifact"""
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
            logging.info(
                f"Data transformation artifact: {data_transformation_artifact}"
            )
            return data_transformation_artifact
        except Exception as e:
            raise SalesException(e, sys)
