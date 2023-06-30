from sales.exception import SalesException
import os, sys
from sales.logger import logging
from sales.pipeline.training_pipeline import TrainPipeline
from fastapi import FastAPI, Request

from sales.utils.main_utils import load_object
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run
from sales.constant.training_pipeline import PREPROCESSOR_OBJECT_DIR
from sales.constant.training_pipeline import SAVED_MODEL_DIR
from sales.constant.application import APP_HOST, APP_PORT
from sales.pipeline.prediction_pipeline import SaleData
from starlette.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response


# app = FastAPI()
# origins = ["*"]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# templates = Jinja2Templates(directory="templates")


# @app.get("/", tags=["authentication"])
# def index():
#     return RedirectResponse(url="/docs")


if __name__ == "__main__":
    training_pipeline = TrainPipeline()
    training_pipeline.run_pipeline()
# @app.get("/train")
# def train_route():
#     try:

#         train_pipeline = TrainPipeline()
#         if train_pipeline.is_pipeline_running:
#             return Response("Training pipeline is already running.")
#         train_pipeline.run_pipeline()
#         return Response("Training successful !!")
#     except Exception as e:
#         return Response(f"Error Occurred! {e}")


# @app.post("/predict")
# def predict_datapoint(request: Request):
#     if request.method == "GET":
#         return templates.TemplateResponse("home.html")
#     else:
#         data = SaleData(
#             item_weight=float(request.form.get("item_weight")),
#             item_fat_content=request.form.get("item_fat_content"),
#             item_visibility=float(request.form.get("item_visibility")),
#             item_type=request.form.get("item_type"),
#             item_mrp=float(request.form.get("item_mrp")),
#             outlet_identifier=request.form.get("outlet_identifier"),
#             outlet_establishment_year=request.form.get("outlet_establishment_year"),
#             outlet_size=request.form.get("outlet_size"),
#             outlet_location_type=request.form.get("outlet_location_type"),
#             outlet_type=request.form.get("outlet_type"),
#         )
#         pred_df = data.get_data_as_data_frame()

#         model_resolver = ModelResolver()
#         if not model_resolver.is_model_exists():
#             return Response("Model is not available")

#         best_model_path = model_resolver.get_best_model_path()
#         model = load_object(file_path=best_model_path)
#         preprocessor = load_object(file_path=PREPROCESSOR_OBJECT_DIR)
#         data_scaled = preprocessor.transform(pred_df)
#         y_pred = model.predict(data_scaled)

#         print(pred_df)

#         return templates.TemplateResponse("home.html", y_pred=y_pred)


# def main():
#     try:
#         training_pipeline = TrainPipeline()
#         training_pipeline.run_pipeline()
#     except Exception as e:
#         print(e)
#         logging.exception(e)


# if __name__ == "__main__":
#     main()

#     app_run(app, host=APP_HOST, port=APP_PORT)
