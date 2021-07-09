#suicide prediction program

# import all modules
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# sklearn modules



# data loading
#enter the location of your input file
input_location = input("Enter your input file location: ")
# check whether the file exists or not
try:
    data = pd.read_csv(input_location)
except:
    print("File not found")
    quit()
# check whether the file is in csv or not
try:
    data.to_csv(input_location)
except:
    print("File not in csv format")
    quit()


# data preprocessing
print(data.info())
print("\n")
print("Some of Data:\n")
print(data.head())
print("\n")


# data Cleaning
total = data.isnull().sum()
precentage = (total/len(data))*100
missing_data = pd.concat([total, precentage], axis=1, keys=['Total', 'Precentage'])
print("Missing Data:\n")
print(missing_data)

# fill missing data with mean value


# drop unnecessary columns
data = data.drop(['Timestamp','comments'], axis=1)
print("\n")   
print("Dataset afterdropping columns:\n")
print(data.head())