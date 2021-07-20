#suicide prediction program

'''
suicide prediction program is working 
            but 
will take time so dont quit in middle
'''

# import all parts as module from src
from src import DataCleaningEncoding
from src import CorrelationMatrix
from src import DataSplitting
from src import FeatureImportance
import src.TuningWithGridSearchCV as gscv
import src.TuningWithRandomizedSearchCV as rscv
from src import AccuracyBarGraph

# ignore all warnings
import warnings
warnings.filterwarnings("ignore")

# import modules
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
import json

# data loading
#enter the location of your input file
input_location = input("Enter your input file location: ")
# check if the file exists
while not os.path.isfile(input_location):
    print("File does not exist")
    exit()
# Check input and read file
if(input_location.endswith(".csv")):
    data = pd.read_csv(input_location)
elif(input_location.endswith(".xlsx")):
    data = pd.read_excel(input_location)
else:
    print("ERROR: File format not supported!")
    exit()

# check data
variable = ['family_size', 'annual_income', 'eating_habits', 
            'addiction_friend', 'addiction', 'medical_history', 
            'depressed', 'anxiety', 'happy_currently', 'suicidal_thoughts']
check = all(item in list(data) for item in variable)
if check is True:
    print("Data is loaded")
else:
    print("Dataset doesnot contain: ", variable)
    exit()

'''
- Data Cleaning nd Encoding
- Corrlation Matrix
- Splitting the data into training and testing
- Feature importance
'''
data = DataCleaningEncoding.dce(data)

CorrelationMatrix.CorrMatrix(data)

X, y, X_train, X_test, y_train, y_test = DataSplitting.DataSplit(data)

FeatureImportance.featuring_importance(X, y)

#Dictionary to store accuracy results of different algorithms
accuracyDict = {}

'''
- Tuning
'''

# Tuning with GridSearchCV
gscv.log_reg_mod_tuning(X_train, X_test, y_train, y_test, accuracyDict)
gscv.tuneKNN(X_train, X_test, y_train, y_test, accuracyDict)
gscv.tuneDT(X_train, X_test, y_train, y_test, accuracyDict)
gscv.tuneRF(X_train, X_test, y_train, y_test, accuracyDict)
gscv.boosting(X_train, X_test, y_train, y_test, accuracyDict)
gscv.bagging(X_train, X_test, y_train, y_test, accuracyDict)
# gscv.stacking(X_train, X_test, y_train, y_test, accuracyDict)

# Tuning with RandomizedSearchCV
rscv.log_reg_mod_tuning(X_train, X_test, y_train, y_test, accuracyDict)
rscv.tuneKNN(X_train, X_test, y_train, y_test, accuracyDict)
rscv.tuneDT(X_train, X_test, y_train, y_test, accuracyDict)
rscv.tuneRF(X_train, X_test, y_train, y_test, accuracyDict)
rscv.boosting(X_train, X_test, y_train, y_test, accuracyDict)
rscv.bagging(X_train, X_test, y_train, y_test, accuracyDict)
# rscv.stacking(X_train, X_test, y_train, y_test, accuracyDict)

print("accuracyDict:\n")
print(json.dumps(accuracyDict, indent=1))

'''
- Accuracy Bar Graph
'''

AccuracyBarGraph.graph(accuracyDict)

'''
- Modelling
'''