import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            pipeline= Pipeline(steps=[("scaler",StandardScaler())])
            preprocessor=ColumnTransformer([("pipeline", pipeline, self.columns)])
            logging.info("Preprocessing object created.")
            return preprocessor
        
        except Exception as e:
            logging.info(CustomException(e,sys))
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            # x_test_provider = Final_Dataset_Provider_Test[['Provider','PotentialFraud']]
            x_train = train_df.drop(columns=['Provider','PotentialFraud'],axis=1)
            y_train = train_df['PotentialFraud']

            x_test = test_df.drop(columns=['Provider','PotentialFraud'],axis=1)
            y_test = test_df['PotentialFraud']

            self.columns = x_train.columns.tolist()

            preprocessing_obj=self.get_data_transformer_object()

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr=preprocessing_obj.fit_transform(x_train)
            input_feature_test_arr=preprocessing_obj.transform(x_test)

            train_arr = np.c_[input_feature_train_arr, np.array(y_train)]
            test_arr = np.c_[input_feature_test_arr, np.array(y_test)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj)
            
            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            logging.info(CustomException(e,sys))