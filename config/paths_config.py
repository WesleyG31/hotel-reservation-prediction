import os

### Data Ingestion Paths ###

raw_dir= "artifacts/raw"

raw_file_path= os.path.join(raw_dir, "raw.csv")
train_file_path = os.path.join (raw_dir,"train.csv")
test_file_path = os.path.join(raw_dir,"test.csv")

config_path= "config/config.yaml"


### Data Preprocessing Paths ###
processed_dir= "artifacts/preprocessed"
processed_train_data_path= os.path.join(processed_dir, "train_data.csv")
processed_test_data_path= os.path.join(processed_dir, "test_data.csv")


### Model Training ####

MODEL_OUTPUT_PATH= "artifacts/model/lgbm_model.pkl"
