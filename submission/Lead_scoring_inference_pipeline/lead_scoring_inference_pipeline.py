##############################################################################
# Import necessary modules
# #############################################################################




import sys
import os
sys.path.insert(0,os.path.abspath(os.path.dirname(__file__)))
import utils, constants
import importlib
importlib.reload(utils)
importlib.reload(constants)
import warnings
warnings.filterwarnings("ignore")



from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta


###############################################################################
# Define default arguments and create an instance of DAG
# ##############################################################################

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2022,7,30),
    'retries' : 1, 
    'retry_delay' : timedelta(seconds=5)
}


Lead_scoring_inference_dag = DAG(
                dag_id = 'Lead_scoring_inference_pipeline',
                default_args = default_args,
                description = 'Inference pipeline of Lead Scoring system',
                schedule_interval = '@hourly',
                catchup = False
)

###############################################################################
# Create a task for encode_data_task() function with task_id 'encoding_categorical_variables'
# ##############################################################################
def encode_features_operation():
    utils.encode_features(db_file_name=constants.DB_FILE_NAME, db_path=constants.DB_PATH, ONE_HOT_ENCODED_FEATURES = constants.ONE_HOT_ENCODED_FEATURES, FEATURES_TO_ENCODE = constants.FEATURES_TO_ENCODE)

encode_features_task = PythonOperator(
        task_id='encoding_categorical_variables',
        provide_context=True,
        python_callable=encode_features_operation,
        dag=Lead_scoring_inference_dag)


###############################################################################
# Create a task for load_model() function with task_id 'generating_models_prediction'
# ##############################################################################
def get_models_prediction_operation():
    utils.get_models_prediction(db_file_name=constants.DB_FILE_NAME, db_path=constants.DB_PATH, model_name = constants.MODEL_NAME, stage = constants.STAGE)

get_models_prediction_task = PythonOperator(
        task_id='generating_models_prediction',
        provide_context=True,
        python_callable=get_models_prediction_operation,
        dag=Lead_scoring_inference_dag)


###############################################################################
# Create a task for prediction_col_check() function with task_id 'checking_model_prediction_ratio'
# ##############################################################################
def prediction_ratio_check_operation():
    utils.prediction_ratio_check(db_file_name=constants.DB_FILE_NAME, db_path=constants.DB_PATH)

prediction_ratio_check_task = PythonOperator(
        task_id='checking_model_prediction_ratio',
        provide_context=True,
        python_callable=prediction_ratio_check_operation,
        dag=Lead_scoring_inference_dag)


###############################################################################
# Create a task for input_features_check() function with task_id 'checking_input_features'
# ##############################################################################
def input_features_check_operation():
    utils.input_features_check(db_file_name=constants.DB_FILE_NAME, db_path=constants.DB_PATH, ONE_HOT_ENCODED_FEATURES = constants.ONE_HOT_ENCODED_FEATURES)

input_features_check_task = PythonOperator(
        task_id='checking_input_features',
        provide_context=True,
        python_callable=input_features_check_operation,
        dag=Lead_scoring_inference_dag)


###############################################################################
# Define relation between tasks
# ##############################################################################
encode_features_task >> input_features_check_task >> get_models_prediction_task >> prediction_ratio_check_task