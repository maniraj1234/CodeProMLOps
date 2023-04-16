'''
filename: utils.py
functions: encode_features, get_train_model
creator: shashank.gupta
version: 1
'''

###############################################################################
# Import necessary modules
# ##############################################################################

import pandas as pd
import numpy as np

import sqlite3
from sqlite3 import Error
import os

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pycaret.classification import *
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

import constants
import warnings
warnings.filterwarnings("ignore")


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from datetime import datetime
from datetime import date
import time
from skopt import BayesSearchCV # run pip install scikit-optimize

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score

from lead_scoring_training_pipeline import constants

###############################################################################
# Define the function to encode features
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
       

    OUTPUT
        1. Save the encoded features in a table - features
        2. Save the target variable in a separate table - target


    SAMPLE USAGE
        encode_features()
        
    **NOTE : You can modify the encode_featues function used in heart disease's inference
        pipeline from the pre-requisite module for this.
    '''
    filename = os.path.join(db_path, db_file_name)

    try:
        conn = sqlite3.connect(filename)
        df = pd.read_sql_query('SELECT * FROM model_input', conn)

        # encoding
        #df[FEATURES_TO_ENCODE] = df[FEATURES_TO_ENCODE])
        # Next, create one-hot-encoded variables, add to dataframe, drop old columns
        df_dummies = pd.get_dummies(df[FEATURES_TO_ENCODE], drop_first=True)
        df = pd.concat([df, df_dummies], axis=1)
        df.drop(FEATURES_TO_ENCODE, axis=1, inplace = True)

        df['app_complete_flag'].to_sql('target', conn, if_exists='replace', index=False)
        df[ONE_HOT_ENCODED_FEATURES].to_sql('features', conn, if_exists='replace', index=False)
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()


###############################################################################
# Define the function to train the model
# ##############################################################################

def get_trained_model(db_file_name, db_path):
    '''
    This function setups mlflow experiment to track the run of the training pipeline. It 
    also trains the model based on the features created in the previous function and 
    logs the train model into mlflow model registry for prediction. The input dataset is split
    into train and test data and the auc score calculated on the test data and
    recorded as a metric in mlflow run.   

    INPUTS
        db_file_name : Name of the database file
        db_path : path where the db file should be


    OUTPUT
        Tracks the run in experiment named 'Lead_Scoring_Training_Pipeline'
        Logs the trained model into mlflow model registry with name 'LightGBM'
        Logs the metrics and parameters into mlflow run
        Calculate auc from the test data and log into mlflow run  

    SAMPLE USAGE
        get_trained_model()
    '''
    filename = os.path.join(db_path, db_file_name)

    try:
        conn = sqlite3.connect(filename)
        y = pd.read_sql_query('SELECT * FROM target', conn)
        X = pd.read_sql_query('SELECT * FROM features', conn)
        
        if('app_complete_flag' in X.columns):
            X.drop(['app_complete_flag'],axis = 1, inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

        mlflow.set_tracking_uri(constants.TRACKING_URI)

        #Model Training
        model_config = {
                'boosting_type': 'gbdt',
                'class_weight': None,
                'colsample_bytree': 1.0,
                'importance_type': 'split' ,
                'learning_rate': 0.1,
                'max_depth': -1,
                'min_child_samples': 20,
                'min_child_weight': 0.001,
                'min_split_gain': 0.0,
                'n_estimators': 100,
                'n_jobs': -1,
                'num_leaves': 31,
                'objective': None,
                'random_state': 40,
                'reg_alpha': 0.0,
                'reg_lambda': 0.0,
                'silent': 'warn',
                'subsample': 1.0,
                'subsample_for_bin': 200000 ,
                'subsample_freq': 0
                }

        with mlflow.start_run(run_name='test') as run:
            #Model Training
            clf = lgb.LGBMClassifier()
            clf.set_params(**model_config) 
            clf.fit(X_train, y_train)

            mlflow.sklearn.log_model(sk_model=clf,artifact_path="models", registered_model_name='LightGBM')
            mlflow.log_params(model_config)    

            # predict the results on training dataset
            y_pred=clf.predict(X_test)

            #Log metrics
            acc=accuracy_score(y_pred, y_test)
            conf_mat = confusion_matrix(y_pred, y_test)
            precision = precision_score(y_pred, y_test,average= 'macro')
            recall = recall_score(y_pred, y_test, average= 'macro')
            f1 = f1_score(y_pred, y_test, average='macro')
            auc = roc_auc_score(y_pred,y_test,average='macro')
            cm = confusion_matrix(y_test, y_pred)
            tn = cm[0][0]
            fn = cm[1][0]
            tp = cm[1][1]
            fp = cm[0][1]
            class_zero = precision_recall_fscore_support(y_test, y_pred, average='binary',pos_label=0)
            class_one = precision_recall_fscore_support(y_test, y_pred, average='binary',pos_label=1)

            mlflow.log_metric('test_accuracy', acc)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("Precision", precision)
            mlflow.log_metric("Recall", recall)
            mlflow.log_metric("Precision_0", class_zero[0])
            mlflow.log_metric("Precision_1", class_one[0])
            mlflow.log_metric("Recall_0", class_zero[1])
            mlflow.log_metric("Recall_1", class_one[1])
            mlflow.log_metric("f1_0", class_zero[2])
            mlflow.log_metric("f1_1", class_one[2])
            mlflow.log_metric("False Negative", fn)
            mlflow.log_metric("True Negative", tn)
            mlflow.log_metric("AUC", auc)

            runID = run.info.run_uuid
            print("Inside MLflow Run with id {}".format(runID))
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()
   
