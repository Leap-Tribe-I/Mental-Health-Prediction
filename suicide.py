#suicide prediction program

# import modules
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os.path

# modules for encoding
from sklearn import preprocessing

# sklearn modules for model creation



# data loading
def main():
    global data
    global input_location
    #enter the location of your input file
    input_location = input("Enter your input file location: ")

    #Check input and read file
    if(input_location.endswith(".csv")):
        data = pd.read_csv(input_location)
    elif(input_location.endswith(".xlsx")):
        data = pd.read_excel(input_location)
    else:
        print("ERROR: File format not supported!")
        main()

main()

# data preprocessing
print(data.info())
print("\n")
print("Some of Data:\n")
print(data.head())
print("\n")


# data Cleaning
# total = data.isnull().sum()
# precentage = (total/len(data))*100
# missing_data = pd.concat([total, precentage], axis=1, keys=['Total', 'Precentage'])
# print("Missing Data:\n")
# print(missing_data)

# # fill missing data with mean value


# # drop unnecessary columns
if 'Timestamp' in data:
    data = data.drop(['Timestamp'], axis=1)
print("\n")   
print("Dataset afterdropping columns:\n")
print(data.head())

# data encoding
labelDictionary = {}
for feature in data:
    le = preprocessing.LabelEncoder()
    le.fit(data[feature])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    data[feature] = le.transform(data[feature])
    # Get labels
    labelKey = 'label_' + feature
    labelValue = [*le_name_mapping]
    labelDictionary[labelKey] =labelValue

for key, value in labelDictionary.items():     
    print(key, value)

print("\n")
print("Dataset after encoding:\n")
print(data.head())
print("\n")

# output the encoded data
data.to_csv(input_location + '_encoded.csv')
print("\n")
print("Encoded data saved as: " + input_location + '_encoded.csv')
