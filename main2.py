from sales.exception import SalesException
import os, sys
from sales.logger import logging
from sales.pipeline.training_pipeline import TrainPipeline
from sales.pipeline.prediction_pipeline import PredictPipeline
from sales.utils.main_utils import load_object

from flask import Flask, request, render_template
from sales.constant.application import APP_HOST, APP_PORT
from sales.pipeline.prediction_pipeline import CustomData
from sales.ml.model.estimator import ModelResolver

# app = Flask(__name__)




# @app.route("/")
# def home_page():
#     return render_template("home.html")


# @app.route("/predict", methods=["GET", "POST"])
# def predict():
#     if request.method == "GET":
#         return render_template("home.html")
#     else:
#         data = CustomData(
#             item_weight=float(request.form.get("item_weight")),
#             item_fat_content=request.form.get("item_fat_content"),
#             item_visibility=request.form.get("item_visibility"),
#             item_type=request.form.get("item_type"),
#             item_mrp=float(request.form.get("item_mrp")),
#             outlet_identifier=request.form.get("outlet_identifier"),
#             outlet_establishment_year=request.form.get("outlet_establishment_year"),
#             outlet_size=request.form.get("outlet_size"),
#             outlet_location_type=request.form.get("outlet_location_type"),
#             outlet_type=request.form.get("outlet_type"),
#         )
        
#         pred_df = data.get_data_as_data_frame()
#         print(request.data)
#         predict_pipeline=PredictPipeline()
#         pred=predict_pipeline.predict(pred_df)
        
#         print(pred_df)
#         print("Before Prediction")
#         result=round(pred[0],2)

#         return render_template("home.html",final_result=result)


# if __name__ == "__main__":
#     app.run(host=APP_HOST,debug=True, port=APP_PORT)
if __name__ == "__main__":
    training_pipeline = TrainPipeline()
    training_pipeline.run_pipeline()