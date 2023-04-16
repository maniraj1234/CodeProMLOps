##############################################################################
# Import necessary modules
# #############################################################################
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

from datetime import datetime, timedelta
import sys
import os
sys.path.insert(0,os.path.abspath(os.path.dirname(__file__)))
import utils
import constants
import importlib
importlib.reload(constants)
importlib.reload(utils)
import warnings
warnings.filterwarnings("ignore")

###############################################################################
# Define default arguments and DAG
# ##############################################################################
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2022,7,30),
    'retries' : 1, 
    'retry_delay' : timedelta(seconds=5)
}


ML_training_dag = DAG(
                dag_id = 'Lead_scoring_training_pipeline',
                default_args = default_args,
                description = 'Training pipeline for Lead Scoring System',
                schedule_interval = '@monthly',
                catchup = False
)

###############################################################################
# Create a task for encode_features() function with task_id 'encoding_categorical_variables'
# ##############################################################################
def encode_features_operation():
    utils.encode_features(db_file_name=constants.DB_FILE_NAME, db_path=constants.DB_PATH, ONE_HOT_ENCODED_FEATURES = constants.ONE_HOT_ENCODED_FEATURES, FEATURES_TO_ENCODE = constants.FEATURES_TO_ENCODE)

encode_features_task = PythonOperator(
    task_id='encoding_categorical_variables',
    provide_context=True,
    python_callable=encode_features_operation,
    dag=ML_training_dag)

###############################################################################
# Create a task for get_trained_model() function with task_id 'training_model'
# ##############################################################################
def get_trained_model_operation():
    utils.get_trained_model(db_file_name=constants.DB_FILE_NAME, db_path=constants.DB_PATH)

get_trained_model_task = PythonOperator(
    task_id='training_model',
    provide_context=True,
    python_callable=get_trained_model_operation,
    dag=ML_training_dag)


###############################################################################
# Define relations between tasks
# ##############################################################################
encode_features_task >> get_trained_model_task
