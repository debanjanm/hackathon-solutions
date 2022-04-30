###############################################################################
# importing required libraries
import os

import pandas as pd
import numpy as np

from helpers import missing_values_table
from helpers import CustomImputer

###############################################################################
path="C:/Users/acer/OneDrive/Documents/GitHub/hackathon-solutions/drivendata/pump_it_up_data_mining_the_water_table"
os.chdir(path)
print(os.getcwd())  # Prints the current working directory

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
data = train_value

# Remove Irrelevant Features
list(data.columns) 

feature_name = 'amount_tsh'
data[feature_name].value_counts()
data[feature_name].nunique()
