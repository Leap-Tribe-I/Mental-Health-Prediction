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
import src.data as dt
import src.split as spl 
import src.feature as ft
import src.tuned_algos as talgo 
import src.eval as evl

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

#calling dataMagic function from src.data to clean and encode the data
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

#calling the datasplit function from src.split to split our data into training and testing sets
#returning the independent and dpendent variables and storing the below
X, y, X_train, X_test, y_train, y_test = spl.datasplit(data)
ft.findFeature(X, y)

#Creating a dictionary to store accuracy score of different algorithms
accuracyDict = {}

#Calling prediction algorithms from src.tuned_algos and passing training data to train the algo and testing
#data to test and rate the accuracy of the algorithm
#calling LogisticRegression
talgo.log_reg_mod_tuning(X_train, X_test, y_train, y_test, accuracyDict)
#K nearest neighbour
talgo.tuneKNN(X_train, X_test, y_train, y_test, accuracyDict)
#Decision tree
talgo.tuneDTree(X_train, X_test, y_train, y_test, accuracyDict)
#Random Forrest
talgo.tuneRF(X_train, X_test, y_train, y_test, accuracyDict)
#boosting with randomizedCV tuning implemented
talgo.boosting_rand(X_train, X_test, y_train, y_test, accuracyDict)
#bagging with randomizedCV tuning implemented
talgo.bagging_rand(X_train, X_test, y_train, y_test, accuracyDict)

#printing the accuracyDict containing accuracy scores of different algorithms
print("accuracyDict:\n")
print(json.dumps(accuracyDict, indent=1))