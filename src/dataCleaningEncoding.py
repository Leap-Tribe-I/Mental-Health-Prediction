# importing basic libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
# modules for encoding
from sklearn import preprocessing

labelDictionary = {}

def dataMagic(data):
    # check data
    variable = ['family_size', 'annual_income', 'eating_habits', 
                'addiction_friend', 'addiction', 'medical_history', 
                'depressed', 'anxiety', 'happy_currently', 'suicidal_thoughts']
    check = all(item in list(data) for item in variable)
    
    #drop unessary columns
    if check is True:
        print("Data is loaded")
    else:
        print("Dataset doesnot contain: ", variable)
        exit()
        
    if 'Timestamp' in data:
        data = data.drop(['Timestamp'], axis=1)
    # print("\n")   
    # print("Dataset afterdropping columns:\n")
    # print(data.head())

    # data encoding
    for feature in data:
        le = preprocessing.LabelEncoder()
        #encoding the data set using the LabelEncoder function 
        le.fit(data[feature])
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        data[feature] = le.transform(data[feature])
        # Get labels
        labelKey = 'label_' + feature
        labelValue = [*le_name_mapping]
        labelDictionary[labelKey] =labelValue
    return data


    # print(labelDictionary)
    # for key, value in labelDictionary.items():     
    #     print(key, value)

    # print("\n")
    # print("Dataset after encoding:\n")
    # print(data.head())
    # print("\n")

    # output the encoded data
    # data.to_csv(input_location + '_encoded.csv')
    # print("\n")
    # print("Encoded data saved as: " + input_location + '_encoded.csv')