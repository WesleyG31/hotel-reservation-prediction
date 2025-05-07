import os
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml, load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE


logger=get_logger(__name__)

class DataProcessor:
    def __init__(self, train_path, test_path, processed_dir, config):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        self.config = read_yaml(config)

        os.makedirs(self.processed_dir, exist_ok=True)

    def preprocess_data(self, df):
        try:
            logger.info("Starting data preprocessing...")

            logger.info("Dropping unnecessary columns...")
            df.drop(columns=["Booking_ID"], inplace=True)

            logger.info("Dropped duplicate rows.")
            df.drop_duplicates(inplace=True)

            logger.info("Categorical and numerical columns...")
            cat_cols = self.config['data_processing']['categorical_columns']
            num_cols = self.config['data_processing']['numerical_columns']

            logger.info("Applying label encoding to categorical columns...")
            
            label_encoder=LabelEncoder()
            mappings={}

            for col in cat_cols:
                df[col]=label_encoder.fit_transform(df[col])
                mappings[col]={label:code for label, code in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}
            
            logger.info("Labels mappings are: ")
            for col, mapping in mappings.items():
                logger.info(f"{col} : {mapping}")

            logger.info("Skewness handling")
            skewness_threshold=self.config['data_processing']['skewness_threshold']
            skewness = df[num_cols].apply(lambda x:x.skew())

            for column in skewness[skewness>skewness_threshold].index:
                df[column]=np.log1p(df[column]) 

            return df
        
        except Exception as e:
            logger.error(f"Error during preprocess {e}")
            raise CustomException("Error while preprocessing")

    def balance_data(self, df):
        try:
            logger.info("Handling Imbalanced data")

            smote=SMOTE()
            X= df.drop(columns=["booking_status"])
            y= df["booking_status"]

            x_res, y_res = smote.fit_resample(X,y)
            df_res=pd.DataFrame(x_res, columns= X.columns)
            df_res["booking_status"]=y_res

            logger.info("Data balanced succesfully")

            return df_res

        except Exception as e:
            logger.error(f"Error while trying to balance the data {e}")   
            raise CustomException("Error while trying to balance the data")   

    def select_features(self,df):
        try:
            logger.info("Stating feature selection")
            
            X=df.drop(columns=["booking_status"])
            y=df["booking_status"]

            logger.info("RandomForest starting...")
            model= RandomForestClassifier()
            model.fit(X,y)
            
            logger.info("Selecting features")
            feature_importance=model.feature_importances_
            feature_importance_df=pd.DataFrame({'feature': X.columns,
                                               'importance': feature_importance})
            top_feature_importance_df=feature_importance_df.sort_values(by="importance", ascending=False)
            num_feature_to_select= self.config['data_processing']['no_of_features']
            top_features=top_feature_importance_df['feature'].head(num_feature_to_select).values
            top_features_df=df[top_features.tolist()+["booking_status"]]
            logger.info("Feature selection completed succesfully")

            return top_features_df
        except Exception as e:
            logger.error(f"Error while feature selection {e}")
            raise CustomException("Error while feature selection",e)
        
    def save_data(self, df, file_path):
        try:
            logger.info(f"Saving data in {file_path}")
            df.to_csv(file_path, index=False)
            
            logger.info(f"Data Saved Succesfully in {file_path}")

        except Exception as e:
            logger.error(f"Error while saving Data {e}")
            raise CustomException("Error while saving Data",e)
        
    def process(self):
        try: 
            logger.info("loading data from RAW directory")

            train_df=load_data(self.train_path)
            test_df=load_data(self.test_path)

            train_df=self.preprocess_data(train_df)
            test_df=self.preprocess_data(test_df)

            train_df=self.balance_data(train_df)
            #test_df=self.balance_data(test_df)

            train_df=self.select_features(train_df)

            test_df=test_df[train_df.columns]

            self.save_data(train_df,processed_train_data_path)
            self.save_data(train_df,processed_test_data_path)

            logger.info("Data processing Completed Successfully")

        except Exception as e:
            logger.error(f"Error while preprocess pipeline {e}")
            raise CustomException("Error while preprocess pipeline",e)
        

if __name__== "__main__":
    processor=DataProcessor(train_file_path,test_file_path,processed_dir,config_path)
    processor.process()