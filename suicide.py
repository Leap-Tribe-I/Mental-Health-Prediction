#suicide prediction program
'''
suicide prediction program is working 
            but 
will take time so dont quit in middle
'''
# import modules
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
import json
import src.data_cleaning_encoding as dt
import src.data_split as spl 
import src.find_feature_variable as ft
import src.grid_searchCV_algos as gcv
import src.model_evaluator as ev
import src.randomized_searchCV_algos as rcv
import src.accuracy_plot as ap
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

#calling dataMagic function from src.data_cleaning_encoding to clean and encode the data
data = dt.dataMagic(data)

#creating a covarinance matrix of the encoded data to visualize correlation between data points
corr = data.corr()
#printing the Covarinance matrix
print("\n")
print("Correlation Matrix:\n")
print(corr)
print("\n")
f, ax = plt.subplots(figsize=(9, 9))
sns.heatmap(corr, vmax=.8, square=True, annot=True)
plt.show()

#calling the datasplit function from src.data_split to split our data into training and testing sets
#returning the independent and dpendent variables and storing the below
X, y, X_train, X_test, y_train, y_test = spl.datasplit(data)
ft.findFeature(X, y)

#Creating a dictionary to store accuracy score of different algorithms
accuracyDict = {}

#Calling prediction algorithms from src.tuned_algos and passing training data to train the algo and testing
#data to test and rate the accuracy of the algorithm
#calling LogisticRegression with GridSearchCV
gcv.log_reg_mod_tuning(X_train, X_test, y_train, y_test, accuracyDict)
#K nearest neighbour with GridSearchCV
gcv.tuneKNN(X_train, X_test, y_train, y_test, accuracyDict)
#Decision tree with GridSearchCV
gcv.tuneDTree(X_train, X_test, y_train, y_test, accuracyDict)
#Random Forrest with GridSearchCV
gcv.tuneRF(X_train, X_test, y_train, y_test, accuracyDict)
# Boosting with GridSearchCV
gcv.boosting(X_train, X_test, y_train, y_test, accuracyDict)
# Boosting with GridSearchCV
gcv.bagging(X_train, X_test, y_train, y_test, accuracyDict)


#calling functions from src.randomized_searchCV_algos and passing training data to train the algo and testing
#data to test and rate the accuracy of the algorithm
# LogisticRegression with randomizedCV
rcv.log_reg_mod_tuning_rand(X_train, X_test, y_train, y_test, accuracyDict)
# K Nearest neighbour with randomizedCV
rcv.tuneKNN_rand(X_train, X_test, y_train, y_test, accuracyDict)
# Decision Tree with randomizedCV
rcv.tuneDTree_rand(X_train, X_test, y_train, y_test, accuracyDict)
# Random Forest with randomizedCV
rcv.tuneRF_rand(X_train, X_test, y_train, y_test, accuracyDict)
# Boosting with randomizedCV
rcv.boosting_rand(X_train, X_test, y_train, y_test, accuracyDict)
# Bagging with randomizedCV tuning implemented
rcv.bagging_rand(X_train, X_test, y_train, y_test, accuracyDict)

#printing the accuracyDict containing accuracy scores of different algorithms
print("accuracyDict:\n")
print(json.dumps(accuracyDict, indent=1))
#calling accuracy_graph function from src.accuracy_plot to plot the accuracy scores of different algorithms
ap.accuracy_graph(accuracyDict)