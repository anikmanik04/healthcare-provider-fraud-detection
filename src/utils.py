import numpy as np
import pandas as pd
import joblib
import os
import pickle
import sys
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import f1_score

def create_feature_using_groupby(df, gruopby_col, operation_col,operation):
    '''
    This function groupby the 'df' dataframe by 'gruopby_col' and performs 'operation' on 'operation_col'
    '''
    
    for col in operation_col:
        # create new column name for the dataframe
        new_col_name = 'Per'+''.join(gruopby_col)+'_'+operation+'_'+col
        #print(new_col_name)
        df[new_col_name] = df.groupby(gruopby_col)[col].transform(operation)
    return df

# define a function to preprocess raw test data
def preprocess_data(Provider, Beneficiary, Inpatient, Outpatient):
    
    # Replacing 2 with 0 for chronic conditions, Zero indicates chronic condition is No
    Beneficiary = Beneficiary.replace({'ChronicCond_Alzheimer': 2, 'ChronicCond_Heartfailure': 2, 'ChronicCond_KidneyDisease': 2,
                               'ChronicCond_Cancer': 2, 'ChronicCond_ObstrPulmonary': 2, 'ChronicCond_Depression': 2, 
                               'ChronicCond_Diabetes': 2, 'ChronicCond_IschemicHeart': 2, 'ChronicCond_Osteoporasis': 2, 
                               'ChronicCond_rheumatoidarthritis': 2, 'ChronicCond_stroke': 2 }, 0)

    # For RenalDiseaseIndicator replacing 'Y' with 1
    Beneficiary = Beneficiary.replace({'RenalDiseaseIndicator': 'Y'}, 1)

    # convert all these columns datatypes to numeric
    Beneficiary[["ChronicCond_Alzheimer", "ChronicCond_Heartfailure", "ChronicCond_KidneyDisease", "ChronicCond_Cancer", "ChronicCond_ObstrPulmonary", "ChronicCond_Depression", "ChronicCond_Diabetes", "ChronicCond_IschemicHeart", "ChronicCond_Osteoporasis", "ChronicCond_rheumatoidarthritis", "ChronicCond_stroke", "RenalDiseaseIndicator"]] = Beneficiary[["ChronicCond_Alzheimer", "ChronicCond_Heartfailure", "ChronicCond_KidneyDisease", "ChronicCond_Cancer", "ChronicCond_ObstrPulmonary", "ChronicCond_Depression", "ChronicCond_Diabetes", "ChronicCond_IschemicHeart", "ChronicCond_Osteoporasis", "ChronicCond_rheumatoidarthritis", "ChronicCond_stroke", "RenalDiseaseIndicator"]].apply(pd.to_numeric)

    # calculate patient risk score by summing up all risk scores
    Beneficiary['Patient_Risk_Score'] = Beneficiary['ChronicCond_Alzheimer'] + Beneficiary['ChronicCond_Heartfailure'] + \
                                            Beneficiary['ChronicCond_KidneyDisease'] + Beneficiary['ChronicCond_Cancer'] +\
                                            Beneficiary['ChronicCond_ObstrPulmonary'] + Beneficiary['ChronicCond_Depression'] +\
                                        Beneficiary['ChronicCond_Diabetes'] + Beneficiary['ChronicCond_IschemicHeart'] +\
                                        Beneficiary['ChronicCond_Osteoporasis'] + Beneficiary['ChronicCond_rheumatoidarthritis'] +\
                                        Beneficiary['ChronicCond_stroke'] + Beneficiary['RenalDiseaseIndicator'] 

    # Replacing '2' with '0' for Gender Type
    Beneficiary = Beneficiary.replace({'Gender': 2}, 0)

    # Convert Date of Birth and Date of Death from String to Datetime format
    Beneficiary['DOB'] = pd.to_datetime(Beneficiary['DOB'] , format = '%Y-%m-%d')
    Beneficiary['DOD'] = pd.to_datetime(Beneficiary['DOD'],format = '%Y-%m-%d')

    # Get the birth month and Birth year for DOB and DOD
    Beneficiary['Birth_Year'] = Beneficiary['DOB'].dt.year
    Beneficiary['Birth_Month'] = Beneficiary['DOB'].dt.month

    Beneficiary['Patient_Age'] = round(((Beneficiary['DOD'] - Beneficiary['DOB']).dt.days)/365)
    Beneficiary.Patient_Age.fillna(round(((pd.to_datetime('2009-12-01',format ='%Y-%m-%d')-Beneficiary['DOB']).dt.days)/365),inplace=True)

    # Set value=1 if the patient is dead i.e DOD value is not null
    Beneficiary['isDead'] = 0
    Beneficiary.loc[Beneficiary.DOD.notna(), 'isDead'] = 1

    # convert ClaimStartDt, ClaimEndDt from string to datetime format
    Inpatient['ClaimStartDt'] = pd.to_datetime(Inpatient['ClaimStartDt'] , format = '%Y-%m-%d')
    Inpatient['ClaimEndDt'] = pd.to_datetime(Inpatient['ClaimEndDt'],format = '%Y-%m-%d')

    # convert AdmissionDt, DischargeDt from string to datetime format
    Inpatient['AdmissionDt'] = pd.to_datetime(Inpatient['AdmissionDt'] , format = '%Y-%m-%d')
    Inpatient['DischargeDt'] = pd.to_datetime(Inpatient['DischargeDt'],format = '%Y-%m-%d')

    # Calculate Hospitalization_Duration = DischargeDt - AdmissionDt
    Inpatient['Hospitalization_Duration'] = ((Inpatient['DischargeDt'] - Inpatient['AdmissionDt']).dt.days)+1
    # Calculate Claim_Period = ClaimEndDt - ClaimStartDt
    Inpatient['Claim_Period'] = ((Inpatient['ClaimEndDt'] - Inpatient['ClaimStartDt']).dt.days)+1

    # ExtraClaimDays = Claim_Period - Hospitalization_Duration
    Inpatient['ExtraClaimDays'] = np.where(Inpatient['Claim_Period']>Inpatient['Hospitalization_Duration'], Inpatient['Claim_Period'] - Inpatient['Hospitalization_Duration'], 0)

    # Get the months and year of claim start and claim end
    Inpatient['ClaimStart_Year'] = Inpatient['ClaimStartDt'].dt.year
    Inpatient['ClaimStart_Month'] = Inpatient['ClaimStartDt'].dt.month
    Inpatient['ClaimEnd_Year'] = Inpatient['ClaimEndDt'].dt.year
    Inpatient['ClaimEnd_Month'] = Inpatient['ClaimEndDt'].dt.month

    # Get the month and year of Admission_Year and Admission_Month
    Inpatient['Admission_Year'] = Inpatient['AdmissionDt'].dt.year
    Inpatient['Admission_Month'] = Inpatient['AdmissionDt'].dt.month

    Inpatient['Discharge_Year'] = Inpatient['DischargeDt'].dt.year
    Inpatient['Discharge_Month'] = Inpatient['DischargeDt'].dt.month

    # convert ClaimStartDt, ClaimEndDt from string to datetime format
    Outpatient['ClaimStartDt'] = pd.to_datetime(Outpatient['ClaimStartDt'] , format = '%Y-%m-%d')
    Outpatient['ClaimEndDt'] = pd.to_datetime(Outpatient['ClaimEndDt'],format = '%Y-%m-%d')

    # Get the months and year of claim start and claim end
    Outpatient['ClaimStart_Year'] = Outpatient['ClaimStartDt'].dt.year
    Outpatient['ClaimStart_Month'] = Outpatient['ClaimStartDt'].dt.month
    Outpatient['ClaimEnd_Year'] = Outpatient['ClaimEndDt'].dt.year
    Outpatient['ClaimEnd_Month'] = Outpatient['ClaimEndDt'].dt.month

    # Calculate Claim_Period = ClaimEndDt - ClaimStartDt
    Outpatient['Claim_Period'] = ((Outpatient['ClaimEndDt'] - Outpatient['ClaimStartDt']).dt.days)+1

    # Create a new column Inpatient_or_Outpatient where Inpatient =1 and Outpatient = 0
    Inpatient['Inpatient_or_Outpatient'] = 1
    Outpatient['Inpatient_or_Outpatient'] = 0

    # Merge inpatient and outpatient dataframes based on common columns
    common_columns_test = [ idx for idx in Outpatient.columns if idx in Inpatient.columns]
    Inpatient_Outpatient_Merge = pd.merge(Inpatient, Outpatient, left_on = common_columns_test, right_on = common_columns_test,how = 'outer')

    # Merge beneficiary details with inpatient and outpatient data
    Inpatient_Outpatient_Beneficiary_Merge = pd.merge(Inpatient_Outpatient_Merge, Beneficiary,
                                                      left_on='BeneID',right_on='BeneID',how='inner')

    Final_Dataset = pd.merge(Inpatient_Outpatient_Beneficiary_Merge, Provider , how = 'inner', on = 'Provider' )

    # create new feature total reimbursement amount for inpatient and outpatient
    Final_Dataset['IP_OP_TotalReimbursementAmt'] = Final_Dataset['IPAnnualReimbursementAmt'] + Final_Dataset['OPAnnualReimbursementAmt']
    # create new feature total deductible amount for inpatient and outpatient
    Final_Dataset['IP_OP_AnnualDeductibleAmt'] = Final_Dataset['IPAnnualDeductibleAmt'] + Final_Dataset['OPAnnualDeductibleAmt']

    # Fill missing results using 0
    Final_Dataset = Final_Dataset.fillna(0).copy()
    
    # group by columns to create feature
    columns = ['InscClaimAmtReimbursed', 'DeductibleAmtPaid', 'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt', 'Patient_Age', 'NoOfMonths_PartACov', 'NoOfMonths_PartBCov', 'Hospitalization_Duration', 'Claim_Period', 'Patient_Risk_Score']
    Final_Dataset =  create_feature_using_groupby(Final_Dataset, ['Provider'], columns, 'mean')

    columns = ['InscClaimAmtReimbursed', 'DeductibleAmtPaid', 'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt', 'Patient_Age', 'Hospitalization_Duration', 'Claim_Period', 'Patient_Risk_Score']
    Final_Dataset =  create_feature_using_groupby(Final_Dataset, ['BeneID'], columns, 'mean')

    columns = ['InscClaimAmtReimbursed', 'DeductibleAmtPaid', 'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt', 'Patient_Age', 'Hospitalization_Duration', 'Claim_Period', 'Patient_Risk_Score']
    Final_Dataset =  create_feature_using_groupby(Final_Dataset, ['AttendingPhysician'], columns, 'mean')

    columns = ['InscClaimAmtReimbursed', 'DeductibleAmtPaid', 'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt', 'Patient_Age', 'Hospitalization_Duration', 'Claim_Period', 'Patient_Risk_Score']
    Final_Dataset =  create_feature_using_groupby(Final_Dataset, ['OperatingPhysician'], columns, 'mean')

    columns = ['InscClaimAmtReimbursed', 'DeductibleAmtPaid', 'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt', 'Patient_Age', 'Hospitalization_Duration', 'Claim_Period', 'Patient_Risk_Score']
    Final_Dataset =  create_feature_using_groupby(Final_Dataset, ['OtherPhysician'], columns, 'mean')

    columns = ['InscClaimAmtReimbursed', 'DeductibleAmtPaid', 'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt', 'Patient_Age', 'Hospitalization_Duration', 'Claim_Period', 'Patient_Risk_Score']
    Final_Dataset =  create_feature_using_groupby(Final_Dataset, ['DiagnosisGroupCode'], columns, 'mean')

    columns = ['InscClaimAmtReimbursed', 'DeductibleAmtPaid', 'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt', 'Patient_Age', 'Hospitalization_Duration', 'Claim_Period', 'Patient_Risk_Score']
    Final_Dataset =  create_feature_using_groupby(Final_Dataset, ['ClmAdmitDiagnosisCode'], columns, 'mean')

    columns = ['InscClaimAmtReimbursed', 'DeductibleAmtPaid', 'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt', 'Patient_Age', 'Hospitalization_Duration', 'Claim_Period', 'Patient_Risk_Score']
    Final_Dataset =  create_feature_using_groupby(Final_Dataset, ['ClmProcedureCode_1'], columns, 'mean')

    columns = ['InscClaimAmtReimbursed', 'DeductibleAmtPaid', 'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt', 'Patient_Age', 'Hospitalization_Duration', 'Claim_Period', 'Patient_Risk_Score']
    Final_Dataset =  create_feature_using_groupby(Final_Dataset, ['ClmProcedureCode_2'], columns, 'mean')

    columns = ['InscClaimAmtReimbursed', 'DeductibleAmtPaid', 'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt', 'Patient_Age', 'Hospitalization_Duration', 'Claim_Period', 'Patient_Risk_Score']
    Final_Dataset =  create_feature_using_groupby(Final_Dataset, ['ClmProcedureCode_3'], columns, 'mean')

    columns = ['InscClaimAmtReimbursed', 'DeductibleAmtPaid', 'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt', 'Patient_Age', 'Hospitalization_Duration', 'Claim_Period', 'Patient_Risk_Score']
    Final_Dataset =  create_feature_using_groupby(Final_Dataset, ['ClmProcedureCode_4'], columns, 'mean')

    columns = ['InscClaimAmtReimbursed', 'DeductibleAmtPaid', 'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt', 'Patient_Age', 'Hospitalization_Duration', 'Claim_Period', 'Patient_Risk_Score']
    Final_Dataset =  create_feature_using_groupby(Final_Dataset, ['ClmProcedureCode_5'], columns, 'mean')

    columns = ['InscClaimAmtReimbursed', 'DeductibleAmtPaid', 'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt', 'Patient_Age', 'Hospitalization_Duration', 'Claim_Period', 'Patient_Risk_Score']
    Final_Dataset =  create_feature_using_groupby(Final_Dataset, ['ClmProcedureCode_6'], columns, 'mean')

    columns = ['InscClaimAmtReimbursed', 'DeductibleAmtPaid', 'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt', 'Patient_Age', 'Hospitalization_Duration', 'Claim_Period', 'Patient_Risk_Score']
    Final_Dataset =  create_feature_using_groupby(Final_Dataset, ['ClmDiagnosisCode_1'], columns, 'mean')

    columns = ['InscClaimAmtReimbursed', 'DeductibleAmtPaid', 'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt', 'Patient_Age', 'Hospitalization_Duration', 'Claim_Period', 'Patient_Risk_Score']
    Final_Dataset =  create_feature_using_groupby(Final_Dataset, ['ClmDiagnosisCode_2'], columns, 'mean')

    columns = ['InscClaimAmtReimbursed', 'DeductibleAmtPaid', 'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt', 'Patient_Age', 'Hospitalization_Duration', 'Claim_Period', 'Patient_Risk_Score']
    Final_Dataset =  create_feature_using_groupby(Final_Dataset, ['ClmDiagnosisCode_3'], columns, 'mean')

    columns = ['InscClaimAmtReimbursed', 'DeductibleAmtPaid', 'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt', 'Patient_Age', 'Hospitalization_Duration', 'Claim_Period', 'Patient_Risk_Score']
    Final_Dataset =  create_feature_using_groupby(Final_Dataset, ['ClmDiagnosisCode_4'], columns, 'mean')

    columns = ['InscClaimAmtReimbursed', 'DeductibleAmtPaid', 'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt', 'Patient_Age', 'Hospitalization_Duration', 'Claim_Period', 'Patient_Risk_Score']
    Final_Dataset =  create_feature_using_groupby(Final_Dataset, ['ClmDiagnosisCode_5'], columns, 'mean')

    columns = ['InscClaimAmtReimbursed', 'DeductibleAmtPaid', 'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt', 'Patient_Age', 'Hospitalization_Duration', 'Claim_Period', 'Patient_Risk_Score']
    Final_Dataset =  create_feature_using_groupby(Final_Dataset, ['ClmDiagnosisCode_6'], columns, 'mean')

    # Count the claims per provider
    Final_Dataset =  create_feature_using_groupby(Final_Dataset, ['Provider'], ['ClaimID'], 'count')

    columns = ['ClaimID']
    grp_by_cols = ['BeneID', 'AttendingPhysician', 'OtherPhysician', 'OperatingPhysician', 'ClmAdmitDiagnosisCode', 'ClmProcedureCode_1',
                   'ClmProcedureCode_2', 'ClmProcedureCode_3', 'ClmProcedureCode_4', 'ClmProcedureCode_5', 'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2',
                   'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6', 'DiagnosisGroupCode']
    for ele in grp_by_cols:
        lst = ['Provider', ele]
        Final_Dataset =  create_feature_using_groupby(Final_Dataset, lst, columns, 'count')

    # remove the columns which are not required
    remove_columns=['BeneID', 'ClaimID', 'ClaimStartDt','ClaimEndDt','AttendingPhysician','OperatingPhysician', 'OtherPhysician',
                    'ClmDiagnosisCode_1','ClmDiagnosisCode_2', 'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4','ClmDiagnosisCode_5',
                    'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7','ClmDiagnosisCode_8', 'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10',
                    'ClmProcedureCode_1', 'ClmProcedureCode_2', 'ClmProcedureCode_3','ClmProcedureCode_4', 'ClmProcedureCode_5',
                    'ClmProcedureCode_6','ClmAdmitDiagnosisCode', 'AdmissionDt','ClaimStart_Year', 'ClaimStart_Year', 'ClaimStart_Month',
                    'ClaimEnd_Year', 'ClaimEnd_Month', 'Admission_Year', 'Admission_Month', 'Discharge_Year', 'Discharge_Month',
                    'DischargeDt', 'DiagnosisGroupCode','DOB', 'DOD','Birth_Year', 'Birth_Month','State', 'County']

    Final_Dataset_FE=Final_Dataset.drop(columns=remove_columns, axis=1)

    # Convert type of Gender and Race to categorical
    Final_Dataset_FE.Gender=Final_Dataset_FE.Gender.astype('category')
    Final_Dataset_FE.Race=Final_Dataset_FE.Race.astype('category')

    # Do one hot encoding for gender and Race
    Final_Dataset_FE=pd.get_dummies(Final_Dataset_FE,columns=['Gender','Race'])

    if "PotentialFraud" in list(Provider.columns):
        Final_Dataset_Provider = Final_Dataset_FE.groupby(['Provider','PotentialFraud'],as_index=False).agg('sum')
        Final_Dataset_Provider.PotentialFraud.replace(['Yes','No'],['1','0'],inplace=True)
        Final_Dataset_Provider.PotentialFraud=Final_Dataset_Provider.PotentialFraud.astype('int64')
        return Final_Dataset_Provider
    else:
        Final_Dataset_Provider  = Final_Dataset_FE.groupby(['Provider'],as_index=False).agg('sum')
        return Final_Dataset_Provider
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        logging.info(CustomException(e, sys))
    
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logging.info(CustomException(e, sys))

def evaluate_models(X_train, y_train, X_test, y_test, models, params, threshold=0.5):
    try:
        report = {}

        model_list = list(models.values())
        models_params_list = list(models.keys())
        for i in range(len(model_list)):
            model = model_list[i]
            param=params[models_params_list[i]]

            gs = GridSearchCV(model, param, cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            
            # model.fit(X_train, y_train)  # Train model
            model.fit(X_train,y_train)

            y_train_pred_proba = model.predict_proba(X_train)[:,1]
            y_train_pred = [1 if p>=threshold else 0 for p in y_train_pred_proba]
            y_test_pred_proba = model.predict_proba(X_test)[:,1]
            y_test_pred = [1 if p>=threshold else 0 for p in y_test_pred_proba]

            train_f1_score = f1_score(y_train, y_train_pred)
            test_f1_score = f1_score(y_test, y_test_pred)

            report[models_params_list[i]] = test_f1_score

        return report

    except Exception as e:
        raise CustomException(e, sys)