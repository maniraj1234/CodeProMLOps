# You can create more variables according to your project. The following are the basic variables that have been provided to you
BASE_PATH = 'c:\\Users\\mmadishetty\\Downloads\\Course\\Semester 3\\MLOps\\Assignment\\Assignment\\01_data_pipeline\\'
DB_PATH =  BASE_PATH + "scripts\\"
DB_FILE_NAME = 'utils_output.db'
UNIT_TEST_DB_FILE_NAME = 'unit_test_cases.db'
DATA_DIRECTORY = BASE_PATH + 'notebooks\\Data\\'
INTERACTION_MAPPING = 'interaction_mapping.csv'
INDEX_COLUMNS_TRAINING = ['created_date', 'city_tier', 'first_platform_c',
       'first_utm_medium_c', 'first_utm_source_c', 'total_leads_droppped', 
       'referred_lead', 'app_complete_flag']
INDEX_COLUMNS_INFERENCE = ['created_date', 'city_tier', 'first_platform_c',
       'first_utm_medium_c', 'first_utm_source_c', 'total_leads_droppped',
       'referred_lead', 'app_complete_flag']
NOT_FEATURES = ['interaction_type']




