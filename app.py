from flask import Flask,request, render_template, jsonify
import numpy as np
import pandas as pd
import pickle
import io

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page
@app.route("/")
def welcome():
    return "Welcome to Healthcare Provider Fraud Detection!"

@app.route('/predictfraud',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return "Welcome to Healthcare Provider Fraud Detection Prediction Page!"
    else:
        # Check if all files and provider are in the request
        if 'Provider' not in request.files or 'Inpatient' not in request.files or 'Outpatient' not in request.files or 'Beneficiary' not in request.files:
            return jsonify({"error": "Please upload Inpatient, Outpatient, and Beneficiary CSV files."}), 400
        
        # Get the provider string from the form data
        provider_id = request.form.get("provider_id")
        if not provider_id:
            return jsonify({"error": "Please provide a 'provider_id' string in the form data."}), 400

        # Read CSV files from the request
        try:
            provider_file = request.files['Provider']
            inpatient_file = request.files['Inpatient']
            outpatient_file = request.files['Outpatient']
            beneficiary_file = request.files['Beneficiary']
            
            # Convert files to pandas dataframes
            provider_df = pd.read_csv(io.StringIO(provider_file.read().decode('utf-8')))
            inpatient_df = pd.read_csv(io.StringIO(inpatient_file.read().decode('utf-8')))
            outpatient_df = pd.read_csv(io.StringIO(outpatient_file.read().decode('utf-8')))
            beneficiary_df = pd.read_csv(io.StringIO(beneficiary_file.read().decode('utf-8')))
        except Exception as e:
            return jsonify({"error": f"Error reading CSV files: {str(e)}"}), 500
        
        try:
            data = CustomData(provider_id, provider_df, inpatient_df, outpatient_df, beneficiary_df)
            data_preprocessed = data.get_preprocessed_data()

            predict_pipeline = PredictPipeline()
            predicted_class_label = predict_pipeline.predict(data_preprocessed)
            return jsonify({"predictions": predicted_class_label})
            # return "running fine"

        except Exception as e:
            return jsonify({"error": f"Error in prediction: {str(e)}"}), 500


if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)        