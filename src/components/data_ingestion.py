import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import preprocess_data
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# from src.components.data_transformation import DataTransformation
# from src.components.data_transformation import DataTransformationConfig

# from src.components.model_trainer import ModelTrainerConfig
# from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    # raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            logging.info('Read the dataset as dataframe started.')
            # Load the train data into dataframe
            Provider = pd.read_csv("notebook/archive/Train-1542865627584.csv")
            Beneficiary = pd.read_csv("notebook/archive/Train_Beneficiarydata-1542865627584.csv")
            Inpatient = pd.read_csv("notebook/archive/Train_Inpatientdata-1542865627584.csv")
            Outpatient = pd.read_csv("notebook/archive/Train_Outpatientdata-1542865627584.csv")

            train_provider, test_provider = train_test_split(Provider, test_size=0.2,random_state=42)

            train_df = preprocess_data(train_provider, Beneficiary, Inpatient, Outpatient)
            test_df = preprocess_data(test_provider, Beneficiary, Inpatient, Outpatient)

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            logging.info('Storing the data locally after basic preprocessing.')

            train_df.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_df.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data Inmgestion is completed and stored locally.")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            logging.info(CustomException(e,sys))
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    # data_transformation=DataTransformation()
    # train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    # modeltrainer=ModelTrainer()
    # print(modeltrainer.initiate_model_trainer(train_arr,test_arr))



