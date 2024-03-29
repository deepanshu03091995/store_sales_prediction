from pymongo.mongo_client import MongoClient
from sales.constant.database import DATABASE_NAME

from sales.constant.env_variable import MONGODB_URL_KEY

from sales.exception import SalesException
import certifi
import os, sys
from dotenv import load_dotenv


class MongoDBClient:
    client = None

    def __init__(self, database_name=DATABASE_NAME) -> None:
        try:

            if MongoDBClient.client is None:
                mongo_db_url = os.getenv(MONGODB_URL_KEY)

                MongoDBClient.client = MongoClient(
                    mongo_db_url, tlsCAFile=certifi.where()
                )
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
        except Exception as e:
            raise SalesException(e, sys)
