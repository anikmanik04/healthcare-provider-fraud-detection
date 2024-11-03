import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            oversample = SMOTE(sampling_strategy=0.25)
            X_tr_os, y_tr_os = oversample.fit_resample(X_train, y_train)
            models = {
                "Random Forest": RandomForestClassifier(),
                "Logistic Regression": LogisticRegression()
            }
            params={
                "Random Forest": {'max_depth': [5, 8, 10, 20],
                                'max_features': ['auto', 'sqrt'],
                                'min_samples_split': [2, 5, 10],
                                'n_estimators': [200, 400]},
                "Logistic Regression": {"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}
                }
            model_report:dict = evaluate_models(X_train=X_tr_os,y_train=y_tr_os,X_test=X_test,y_test=y_test,
                                             models=models,params=params)
            
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            logging.info(f"Best model is {best_model_name} and corresponding f1 score is {best_model_score}")

            if best_model_score<0.4:
                logging.info(CustomException("No best model found"))
            else:
                logging.info(f"Best found model on both training and testing dataset")

            best_model = models[best_model_name]
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            predicted = best_model.predict(X_test)

            test_f1_score = f1_score(y_test, predicted)
            return test_f1_score

        except Exception as e:
            logging.info(CustomException(e,sys))