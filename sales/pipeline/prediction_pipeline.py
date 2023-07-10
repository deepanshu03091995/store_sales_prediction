import sys
import pandas as pd
from sales.exception import SalesException
from sales.utils.main_utils import load_object
from sales.constant.training_pipeline import PREPROCESSOR_OBJECT_DIR
from sales.constant.training_pipeline import SAVED_MODEL_DIR



class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            
            model = load_object(file_path=SAVED_MODEL_DIR)
            preprocessor = load_object(file_path=PREPROCESSOR_OBJECT_DIR)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise SalesException(e, sys)


class CustomData:
    def __init__(
        self,
        item_weight: float,
        item_fat_content: str,
        item_visibility: float,
        item_type: str,
        item_mrp: int,
        outlet_identifier: str,
        outlet_establishment_year: int,
        outlet_size: str,
        outlet_location_type: str,
        outlet_type: str,
    ):

        self.item_weight = item_weight
        self.item_fat_content = item_fat_content
        self.item_visibility = item_visibility
        self.item_type = item_type
        self.item_mrp = item_mrp
        self.outlet_identifier = outlet_identifier
        self.outlet_establishment_year = outlet_establishment_year
        self.outlet_size = outlet_size
        self.outlet_location_type = outlet_location_type
        self.outlet_type = outlet_type

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "item_weight": [self.item_weight],
                "item_fat_content": [self.item_fat_content],
                "item_visibility": [self.item_visibility],
                "item_type": [self.item_type],
                "item_mrp": [self.item_mrp],
                "outlet_identifier": [self.outlet_identifier],
                "outlet_establishment_year": [self.outlet_establishment_year],
                "outlet_size": [self.outlet_size],
                "outlet_location_type": [self.outlet_location_type],
                "outlet_type": [self.outlet_type]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise SalesException(e, sys)
