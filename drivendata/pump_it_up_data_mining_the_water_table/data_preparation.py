###############################################################################
###############################################################################
# importing required libraries
import pandas as pd
import numpy as np
import os

#import ppscore as pps

path="C:/Users/acer/OneDrive/Documents/GitHub/hackathon/01-drivendata/Pump it Up Data Mining the Water Table"
os.chdir(path)
print(os.getcwd())  # Prints the current working directory

###############################################################################
#
DATA_DIR = "data" # indicate magical constansts (maybe rather put it on the top of the script)
train_value = pd.read_csv(os.path.join(DATA_DIR, "Training set values.csv"))
train_label = pd.read_csv(os.path.join(DATA_DIR, "Training set labels.csv"))
test_value  = pd.read_csv(os.path.join(DATA_DIR, "Test set values.csv"))

###############################################################################
#2.1.Analyze Data:
    
#2.1.1.Descriptive Statitics    
train_value.info()

train_value.head()

train_value.describe()

train_value.shape

test_value.shape

###############################################################################
#2.2. Data Visualization 

#pps_matrix = pps.matrix(train_value)

#matrix_df = pps_matrix.pivot(columns='x', index='y',  values='ppscore')

###############################################################################
###############################################################################
data = train_value

# Remove Irrelevant Features
list(data.columns) 

feature_name = 'amount_tsh'
data[feature_name].value_counts()
data[feature_name].nunique()

data.shape

columns = ['id','date_recorded','wpt_name','num_private','recorded_by','gps_height','longitude','latitude',
           'extraction_type','extraction_type_group','management','payment_type','quality_group',
            'quantity_group','source_type','source_class']
data.drop(columns, inplace=True, axis=1)

data.shape
###############################################################################
###############################################################################

# Missing Data Analysis
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns

train_missing = missing_values_table(data)
train_missing

# Deletion: Delete Entire Column
columns = ['scheme_name']
data.drop(columns, inplace=True, axis=1)

# Imputation
from sklearn.impute import SimpleImputer
from sklearn.base import TransformerMixin

class CustomImputer(TransformerMixin):
    def __init__(self, cols=None, strategy='mean'):
        self.cols = cols
        self.strategy = strategy

    def transform(self, df):
        X = df.copy()
        impute = SimpleImputer(missing_values=np.nan,strategy=self.strategy)
        if self.cols == None:
                self.cols = list(X.columns)
        for col in self.cols:
                if X[col].dtype == np.dtype('O') : 
                        X[col].fillna(X[col].value_counts().index[0], inplace=True)
                else : X[col] = impute.fit_transform(X[[col]])

        return X

    def fit(self, *_):
        return self

call = CustomImputer()
imputed_data = call.fit_transform(data)   

train_missing = missing_values_table(imputed_data)
train_missing

###############################################################################
###############################################################################
# Feature Encoding
