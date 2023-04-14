##############################################################################
# Import the necessary modules
##############################################################################
import utils
import sqlite3
import constants
import pandas as pd
import os
import city_tier_mapping
import significant_categorical_level


###############################################################################
# Write test cases for load_data_into_db() function
# ##############################################################################

def test_load_data_into_db(DB_FILE_NAME, DB_PATH, UNIT_TEST_DB_FILE_NAME):
    """_summary_
    This function checks if the load_data_into_db function is working properly by
    comparing its output with test cases provided in the db in a table named
    'loaded_data_test_case'

    INPUTS
        DB_FILE_NAME : Name of the database file 'utils_output.db'
        DB_PATH : path where the db file should be present
        UNIT_TEST_DB_FILE_NAME: Name of the test database file 'unit_test_cases.db'

    SAMPLE USAGE
        output=test_get_data()

    """
    try:
        utils.load_data_into_db(DB_FILE_NAME, DB_PATH, constants.DATA_DIRECTORY)
        
        print("initiating db connection")
        conn = sqlite3.connect(os.path.join(DB_PATH, DB_FILE_NAME))
        print("initiating unit test db connection")
        conn_test_db = sqlite3.connect(os.path.join(DB_PATH, UNIT_TEST_DB_FILE_NAME))

        # Load the output data from the actual database
        output_data = pd.read_sql_query("SELECT * FROM loaded_data", con=conn)

        # Load the test data from the test database
        test_data = pd.read_sql_query("SELECT * FROM loaded_data_test_case", con=conn_test_db)
 
        # Compare the two dataframes
        #assert test_data.equals(output_data)
        pd.testing.assert_frame_equal(test_data, output_data)
        print("Test passed")
    except Exception as e:
        print("Test failed")
        print(e)
    finally:
        conn.close()
        conn_test_db.close()

    
    

###############################################################################
# Write test cases for map_city_tier() function
# ##############################################################################
def test_map_city_tier(DB_FILE_NAME, DB_PATH, UNIT_TEST_DB_FILE_NAME):
    """_summary_
    This function checks if map_city_tier function is working properly by
    comparing its output with test cases provided in the db in a table named
    'city_tier_mapped_test_case'

    INPUTS
        DB_FILE_NAME : Name of the database file 'utils_output.db'
        DB_PATH : path where the db file should be present
        UNIT_TEST_DB_FILE_NAME: Name of the test database file 'unit_test_cases.db'

    SAMPLE USAGE
        output=test_map_city_tier()

    """
    try:
        utils.map_city_tier(DB_FILE_NAME, DB_PATH, city_tier_mapping.city_tier_mapping)
        
        print("initiating db connection")
        conn = sqlite3.connect(os.path.join(DB_PATH, DB_FILE_NAME))
        print("initiating unit test db connection")
        conn_test_db = sqlite3.connect(os.path.join(DB_PATH, UNIT_TEST_DB_FILE_NAME))

        # Load the output data from the actual database
        output_data = pd.read_sql_query("SELECT * FROM city_tier_mapped", con=conn)

        # Load the test data from the test database
        test_data = pd.read_sql_query("SELECT * FROM city_tier_mapped_test_case", con=conn_test_db)
 
        # Compare the two dataframes
        #assert test_data.equals(output_data)
        pd.testing.assert_frame_equal(test_data, output_data)
        print("Test passed")
    except Exception as e:
        print("Test failed")
        print(e)
    finally:
        conn.close()
        conn_test_db.close()
    
    
###############################################################################
# Write test cases for map_categorical_vars() function
# ##############################################################################    
def test_map_categorical_vars(DB_FILE_NAME, DB_PATH, UNIT_TEST_DB_FILE_NAME):
    """_summary_
    This function checks if map_cat_vars function is working properly by
    comparing its output with test cases provided in the db in a table named
    'categorical_variables_mapped_test_case'

    INPUTS
        DB_FILE_NAME : Name of the database file 'utils_output.db'
        DB_PATH : path where the db file should be present
        UNIT_TEST_DB_FILE_NAME: Name of the test database file 'unit_test_cases.db'
    
    SAMPLE USAGE
        output=test_map_cat_vars()

    """
    try:
        list_platform = significant_categorical_level.list_platform
        list_medium = significant_categorical_level.list_medium
        list_source = significant_categorical_level.list_source
        utils.map_categorical_vars(DB_FILE_NAME, DB_PATH ,list_platform ,list_medium ,list_source)
        
        print("initiating db connection")
        conn = sqlite3.connect(os.path.join(DB_PATH, DB_FILE_NAME))
        print("initiating unit test db connection")
        conn_test_db = sqlite3.connect(os.path.join(DB_PATH, UNIT_TEST_DB_FILE_NAME))

        # Load the output data from the actual database
        output_data = pd.read_sql_query("SELECT * FROM categorical_variables_mapped", con=conn)

        # Load the test data from the test database
        test_data = pd.read_sql_query("SELECT * FROM categorical_variables_mapped_test_case", con=conn_test_db)
 
        # Compare the two dataframes
        #assert test_data.equals(output_data)
        pd.testing.assert_frame_equal(test_data, output_data)
        print("Test passed")
    except Exception as e:
        print("Test failed")
        print(e)
    finally:
        conn.close()
        conn_test_db.close()    
    

###############################################################################
# Write test cases for interactions_mapping() function
# ##############################################################################    
def test_interactions_mapping():
    """_summary_
    This function checks if test_column_mapping function is working properly by
    comparing its output with test cases provided in the db in a table named
    'interactions_mapped_test_case'

    INPUTS
        DB_FILE_NAME : Name of the database file 'utils_output.db'
        DB_PATH : path where the db file should be present
        UNIT_TEST_DB_FILE_NAME: Name of the test database file 'unit_test_cases.db'

    SAMPLE USAGE
        output=test_column_mapping()

    """ 
   
