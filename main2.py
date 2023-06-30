from sales.exception import SalesException
import os, sys
from sales.logger import logging
from sales.pipeline.training_pipeline import TrainPipeline

from sales.utils.main_utils import load_object

from flask import Flask, request, render_template
from sales.constant.training_pipeline import PREPROCESSOR_OBJECT_DIR
from sales.constant.training_pipeline import SAVED_MODEL_DIR
from sales.constant.application import APP_HOST, APP_PORT
from sales.pipeline.prediction_pipeline import SaleData
from sales.ml.model.estimator import ModelResolver

application = Flask(__name__)

app = application


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        data = SaleData(
            item_weight=float(request.form.get("item_weight")),
            item_fat_content=request.form.get("item_fat_content"),
            item_visibility=float(request.form.get("item_visibility")),
            item_type=request.form.get("item_type"),
            item_mrp=float(request.form.get("item_mrp")),
            outlet_identifier=request.form.get("outlet_identifier"),
            outlet_establishment_year=request.form.get("outlet_establishment_year"),
            outlet_size=request.form.get("outlet_size"),
            outlet_location_type=request.form.get("outlet_location_type"),
            outlet_type=request.form.get("outlet_type"),
        )
        pred_df = data.get_data_as_data_frame()

        model_resolver = ModelResolver()
        if not model_resolver.is_model_exists():
            return Response("Model is not available")

        best_model_path = model_resolver.get_best_model_path()
        model = load_object(file_path=best_model_path)
        preprocessor = load_object(file_path=PREPROCESSOR_OBJECT_DIR)
        data_scaled = preprocessor.transform(pred_df)
        y_pred = model.predict(data_scaled)

        print(pred_df)
        print("Before Prediction")

        return render_template("home.html", y_pred=y_pred[0])


if __name__ == "__main__":
    app.run(host=APP_HOST, port=APP_PORT)
