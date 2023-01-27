from sales.exception import SalesException
import os, sys
from sales.logger import logging
from sales.pipeline.training_pipeline import TrainPipeline

if __name__ == "__main__":
    training_pipeline = TrainPipeline()
    training_pipeline.run_pipeline()
