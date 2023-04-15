"""
Import necessary modules
############################################################################## 
"""

import pandas as pd
import os
import constants
import sqlite3
from sqlite3 import Error
from os.path import isfile

###############################################################################
# Define function to validate raw data's schema
############################################################################### 
def raw_data_schema_check(DATA_DIRECTORY, raw_data_schema):
    '''
    This function check if all the columns mentioned in schema.py are present in
    leadscoring.csv file or not.

   
    INPUTS
        DATA_DIRECTORY : path of the directory where 'leadscoring.csv' 
                        file is present
        raw_data_schema : schema of raw data in the form oa list/tuple as present 
                          in 'schema.py'

    OUTPUT
        If the schema is in line then prints 
        'Raw datas schema is in line with the schema present in schema.py' 
        else prints
        'Raw datas schema is NOT in line with the schema present in schema.py'

    
    SAMPLE USAGE
        raw_data_schema_check
    '''
    filename = os.path.join(DATA_DIRECTORY, 'leadscoring.csv')
    try:
        df = pd.read_csv(os.path.join(DATA_DIRECTORY, 'leadscoring.csv'), index_col=0)
        
        if are_equal_arrays(df.columns, raw_data_schema):
            print('Raw datas schema is in line with the schema present in schema.py')
        else:
            print('Raw datas schema is NOT in line with the schema present in schema.py')
    except Error as e:
        print(e)
   

###############################################################################
# Define function to validate model's input schema
############################################################################### 

def model_input_schema_check(DB_FILE_NAME, DB_PATH, model_input_schema):
    '''
    This function check if all the columns mentioned in model_input_schema in 
    schema.py are present in table named in 'model_input' in db file.

   
    INPUTS
        DB_FILE_NAME : Name of the database file
        DB_PATH : path where the db file should be present
        model_input_schema : schema of models input data in the form oa list/tuple
                          present as in 'schema.py'

    OUTPUT
        If the schema is in line then prints 
        'Models input schema is in line with the schema present in schema.py'
        else prints
        'Models input schema is NOT in line with the schema present in schema.py'
    
    SAMPLE USAGE
        raw_data_schema_check
    '''
    filename = os.path.join(DB_PATH, DB_FILE_NAME)
    try:
        conn = sqlite3.connect(filename)
        df = pd.read_sql_query('SELECT * FROM model_input', conn)
        if are_equal_arrays(df.columns, model_input_schema):
            print('Models input schema is in line with the schema present in schema.py')
        else:
            print('Models input schema is NOT in line with the schema present in schema.py')
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

    
    
