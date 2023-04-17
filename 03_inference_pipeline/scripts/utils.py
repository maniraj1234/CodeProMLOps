'''
filename: utils.py
functions: encode_features, load_model
creator: shashank.gupta
version: 1
'''

###############################################################################
# Import necessary modules
# ##############################################################################

import mlflow
import mlflow.sklearn
import pandas as pd

import sqlite3

import os
import logging

from datetime import datetime
import constants
from sqlite3 import Error

###############################################################################
# Define the function to train the model
# ##############################################################################


def encode_features(db_file_name, db_path, ONE_HOT_ENCODED_FEATURES, FEATURES_TO_ENCODE):
    '''
    This function one hot encodes the categorical features present in our  
    training dataset. This encoding is needed for feeding categorical data 
    to many scikit-learn models.

    INPUTS
        db_file_name : Name of the database file 
        db_path : path where the db file should be
        ONE_HOT_ENCODED_FEATURES : list of the features that needs to be there in the final encoded dataframe
        FEATURES_TO_ENCODE: list of features  from cleaned data that need to be one-hot encoded
        **NOTE : You can modify the encode_featues function used in heart disease's inference
        pipeline for this.

    OUTPUT
        1. Save the encoded features in a table - features

    SAMPLE USAGE
        encode_features()
    '''
    filename = os.path.join(db_path, db_file_name)

    try:
        conn = sqlite3.connect(filename)
        df = pd.read_sql_query('SELECT * FROM model_input', conn)

        # encoding
        # Next, create one-hot-encoded variables, add to dataframe, drop old columns
        df_dummies = pd.get_dummies(df[FEATURES_TO_ENCODE], drop_first=True)
        df = pd.concat([df, df_dummies], axis=1)
        df.drop(FEATURES_TO_ENCODE, axis=1, inplace = True)

        df[ONE_HOT_ENCODED_FEATURES].to_sql('features', conn, if_exists='replace', index=False)
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()

###############################################################################
# Define the function to load the model from mlflow model registry
# ##############################################################################

def get_models_prediction(db_file_name, db_path, model_name, stage):
    '''
    This function loads the model which is in production from mlflow registry and 
    uses it to do prediction on the input dataset. Please note this function will the load
    the latest version of the model present in the production stage. 

    INPUTS
        db_file_name : Name of the database file
        db_path : path where the db file should be
        model  from mlflow model registry
        model name: name of the model to be loaded
        stage: stage from which the model needs to be loaded i.e. production


    OUTPUT
        Store the predicted values along with input data into a table

    SAMPLE USAGE
        load_model()
    '''
    filename = os.path.join(db_path, db_file_name)

    try:
        conn = sqlite3.connect(filename)
        mlflow.set_tracking_uri(constants.TRACKING_URI)

        X = pd.read_sql_query('SELECT * FROM features', conn)
        
        
        logged_model = "models:/LightGBM/Production"
        print(logged_model)
        # Load model as a PyFuncModel.
        loaded_model = mlflow.sklearn.load_model(logged_model)
        # Predict on a Pandas DataFrame.
        # predictions_proba = loaded_model.predict_proba(pd.DataFrame(X))
        print(X.columns)
        predictions = loaded_model.predict(pd.DataFrame(X))
        pred_df = X.copy()
        
        pred_df['app_complete_flag'] = predictions
        # pred_df[["Prob of Not Churn","Prob of Churn"]] = predictions_proba
        # index_msno_mapping = pd.read_sql('select * from index_msno_mapping', cnx)
        # pred_df['index_for_map'] = pred_df.index
        # final_pred_df = pred_df.merge(index_msno_mapping, on='index_for_map') 
        # final_pred_df.to_sql(name='predictions', con=cnx,if_exists='replace',index=False)
        pred_df.to_sql(name='Final_Predictions', con=conn,if_exists='replace',index=False)
        print (pd.DataFrame(predictions,columns=["app_complete_flag"]).head()) 
        # pd.DataFrame(predictions,columns=["Prob of Not Churn","Prob of Churn"]).to_sql(name='Final_Predictions', con=cnx,if_exists='replace',index=False)
        return "Predictions are done and save in Final_Predictions Table"
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()


###############################################################################
# Define the function to check the distribution of output column
# ##############################################################################

def prediction_ratio_check(db_file_name, db_path):
    '''
    This function calculates the % of 1 and 0 predicted by the model and  
    and writes it to a file named 'prediction_distribution.txt'.This file 
    should be created in the ~/airflow/dags/Lead_scoring_inference_pipeline 
    folder. 
    This helps us to monitor if there is any drift observed in the predictions 
    from our model at an overall level. This would determine our decision on 
    when to retrain our model.
    

    INPUTS
        db_file_name : Name of the database file
        db_path : path where the db file should be

    OUTPUT
        Write the output of the monitoring check in prediction_distribution.txt with 
        timestamp.

    SAMPLE USAGE
        prediction_col_check()
    '''
    filename = os.path.join(db_path, db_file_name)

    try:
        conn = sqlite3.connect(filename)

        final_predictions = pd.read_sql_query('SELECT * FROM Final_Predictions', conn)
        final_predictions_count1 = final_predictions[final_predictions['app_complete_flag'] == 1].count()
        final_predictions_count0 = final_predictions[final_predictions['app_complete_flag'] == 0].count()
        final_predictions_totalcount = final_predictions['app_complete_flag'].count()
        final_predictions_count1_percentage = (final_predictions_count1 *100) /final_predictions_totalcount
        final_predictions_count0_percentage = (final_predictions_count0 *100) /final_predictions_totalcount
        # Writing output to a file with timestamp
        with open(f"{constants.FILE_PATH}prediction_distribution.txt", "w") as f:
            f.write(f"zeros_percentage: {final_predictions_count0_percentage}, ones_percentage: {final_predictions_count1_percentage}\n")

        return "Successfully updated prediction distribution file"
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()
###############################################################################
# Define the function to check the columns of input features
# ##############################################################################
   

def input_features_check(db_file_name, db_path, ONE_HOT_ENCODED_FEATURES):
    '''
    This function checks whether all the input columns are present in our new
    data. This ensures the prediction pipeline doesn't break because of change in
    columns in input data.

    INPUTS
        db_file_name : Name of the database file
        db_path : path where the db file should be
        ONE_HOT_ENCODED_FEATURES: List of all the features which need to be present
        in our input data.

    OUTPUT
        It writes the output in a log file based on whether all the columns are present
        or not.
        1. If all the input columns are present then it logs - 'All the models input are present'
        2. Else it logs 'Some of the models inputs are missing'

    SAMPLE USAGE
        input_col_check()
    '''
    filename = os.path.join(db_path, db_file_name)
    try:
        conn = sqlite3.connect(filename)
        df = pd.read_sql_query('SELECT * FROM features', conn)
        if are_equal_arrays(df.columns, ONE_HOT_ENCODED_FEATURES):
            print('All the models input are present')
        else:
            print('Some of the models inputs are missing')
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()
    

def are_equal_arrays(array1, array2):
    '''
    This function check if two arrays are equal or not.

   
    INPUTS
        array1: first array to compare.
        array2: second array to compare.

    OUTPUT
        If both are equal, returns true, otherwise false with debug info printed.
    
    SAMPLE USAGE
        raw_data_schema_check
    '''
    if len(array2) != len(array1):
        print('Length doesnt match left: ' + str(len(array1)) + " right: " + str(len(array2)))
        print("left "+ str(array1))
        print("\n right "+ str(array2))
        return False
    for element in array1:
        if element not in array2:
            print('Element is not matching: '+element)
            return False
    return True