import os
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml

logger= get_logger(__name__)

class DataIngestion:
    def __init__(self,config):
        self.config=config["data_ingestion"]
        self.bucket_name=self.config["bucket_name"]
        self.bucket_file_name=self.config["bucket_file_name"]
        self.train_ratio=self.config["train_ratio"]

        os.makedirs(raw_dir,exist_ok=True)

        logger.info(f"Data Ingestion started with {self.bucket_name} and file is {self.bucket_file_name}")
    
    def download_csv_from_gcp(self):
        try: 
            client= storage.Client()
            bucket=client.bucket(self.bucket_name)
            blob=bucket.blob(self.bucket_file_name)
            
            blob.download_to_filename(raw_file_path)

            logger.info(f"File downloaded from GCP bucket {self.bucket_name} to {raw_file_path}")
            
        except Exception as e:
            logger.error("Error downloading file from GCP bucket")
            raise CustomException("Failed to download csv file", e)
        
    def split_data(self):
        try: 
            logger.info("Splitting data into train and test sets")
            df=pd.read_csv(raw_file_path)

            train_data, test_data=train_test_split(df, test_size=1-self.train_ratio, random_state=42)

            train_data.to_csv(train_file_path, index=False)
            test_data.to_csv(test_file_path, index=False)
            logger.info(f"Train and test data saved to {train_file_path} and {test_file_path} respectively")
        
        except Exception as e:
            logger.error("Error splitting data into train and test sets")
            raise CustomException("Failed to split data", e)
        
    def run(self):
        try:
            logger.info("Starting data ingestion process")
            self.download_csv_from_gcp()
            self.split_data()

            logger.info("Data ingestion process completed successfully")

        except CustomException as ce:
            logger.error(f"CustomException: {str(ce)}")
        
        finally:
            logger.info("Data ingestion process finished")

if __name__=="__main__":
    
    data_ingestion=DataIngestion(read_yaml(config_path))
    data_ingestion.run()

