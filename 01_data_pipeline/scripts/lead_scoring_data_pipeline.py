##############################################################################
# Import necessary modules
# #############################################################################


from airflow import DAG
from airflow.operators.python import PythonOperator

from datetime import datetime, timedelta
import utils
import constants
from mapping import city_tier_mapping
from mapping import significant_categorical_level
import data_validation_checks
import schema
import importlib
importlib.reload(data_validation_checks)
import warnings
warnings.filterwarnings("ignore")


DATA_DIRECTORY = constants.DATA_DIRECTORY
DB_FILE_NAME = constants.DB_FILE_NAME
DB_PATH = constants.DB_PATH
DATA_DIRECTORY = constants.DATA_DIRECTORY
city_tier_mapping = city_tier_mapping.city_tier_mapping
list_platform = significant_categorical_level.list_platform
list_medium = significant_categorical_level.list_medium
list_source = significant_categorical_level.list_source

INTERACTION_MAPPING = constants.INTERACTION_MAPPING
INDEX_COLUMNS_TRAINING = constants.INDEX_COLUMNS_TRAINING
INDEX_COLUMNS_INFERENCE = constants.INDEX_COLUMNS_INFERENCE
NOT_FEATURES = constants.NOT_FEATURES


###############################################################################
# Define default arguments and DAG
###############################################################################

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2022,7,30),
    'retries' : 1, 
    'retry_delay' : timedelta(seconds=5)
}


ML_data_cleaning_dag = DAG(
                dag_id = 'Lead_Scoring_Data_Engineering_Pipeline',
                default_args = default_args,
                description = 'DAG to run data pipeline for lead scoring',
                schedule_interval = '@daily',
                catchup = False
)

###############################################################################
# Create a task for build_dbs() function with task_id 'building_db'
###############################################################################
def build_dbs_operation():
    utils.build_dbs(DB_FILE_NAME, DB_PATH)

build_dbs_task = PythonOperator(
    task_id='building_db',
    provide_context=True,
    python_callable=build_dbs_operation,
    dag=ML_data_cleaning_dag)

###############################################################################
# Create a task for raw_data_schema_check() function with task_id 'checking_raw_data_schema'
###############################################################################

def raw_data_schema_check_operation():
    data_validation_checks.raw_data_schema_check(DATA_DIRECTORY, schema.raw_data_schema)

raw_data_schema_check_task = PythonOperator(
    task_id='checking_raw_data_schema',
    provide_context=True,
    python_callable=raw_data_schema_check_operation,
    dag=ML_data_cleaning_dag)
###############################################################################
# Create a task for load_data_into_db() function with task_id 'loading_data'
##############################################################################

def load_data_into_db_operation():
    utils.load_data_into_db(DB_FILE_NAME, DB_PATH, DATA_DIRECTORY)

load_data_into_db_task = PythonOperator(
    task_id='loading_data',
    provide_context=True,
    python_callable=load_data_into_db_operation,
    dag=ML_data_cleaning_dag)

###############################################################################
# Create a task for map_city_tier() function with task_id 'mapping_city_tier'
###############################################################################

def map_city_tier_operation():
    utils.map_city_tier(DB_FILE_NAME, DB_PATH, city_tier_mapping)

map_city_tier_task = PythonOperator(
    task_id='mapping_city_tier',
    provide_context=True,
    python_callable=map_city_tier_operation,
    dag=ML_data_cleaning_dag)

###############################################################################
# Create a task for map_categorical_vars() function with task_id 'mapping_categorical_vars'
###############################################################################

def map_categorical_vars_operation():
    utils.map_categorical_vars(DB_FILE_NAME, DB_PATH ,list_platform ,list_medium ,list_source)

map_categorical_vars_task = PythonOperator(
    task_id='mapping_categorical_vars',
    provide_context=True,
    python_callable=map_categorical_vars_operation,
    dag=ML_data_cleaning_dag)

###############################################################################
# Create a task for interactions_mapping() function with task_id 'mapping_interactions'
###############################################################################

def interactions_mapping_operation():
    utils.interactions_mapping(DB_FILE_NAME, DB_PATH, INTERACTION_MAPPING, INDEX_COLUMNS_TRAINING, INDEX_COLUMNS_INFERENCE, NOT_FEATURES)

interactions_mapping_task = PythonOperator(
    task_id='mapping_interactions',
    provide_context=True,
    python_callable=interactions_mapping_operation,
    dag=ML_data_cleaning_dag)

###############################################################################
# Create a task for model_input_schema_check() function with task_id 'checking_model_inputs_schema'
###############################################################################

def model_input_schema_check_operation():
    data_validation_checks.model_input_schema_check(DB_FILE_NAME=constants.DB_FILE_NAME, DB_PATH=constants.DB_PATH, model_input_schema=schema.model_input_schema)

model_input_schema_check_task = PythonOperator(
    task_id='checking_model_inputs_schema',
    provide_context=True,
    python_callable=model_input_schema_check_operation,
    dag=ML_data_cleaning_dag)

###############################################################################
# Define the relation between the tasks
###############################################################################

build_dbs_task >> raw_data_schema_check_task >> load_data_into_db_task >> map_city_tier_task >> map_categorical_vars_task >> interactions_mapping_task >> model_input_schema_check_task
