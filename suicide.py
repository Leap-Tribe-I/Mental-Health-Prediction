#suicide prediction program

# import modules
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os.path
import seaborn as sns
#!python suicide.py
# modules for encoding
from sklearn import preprocessing
#modules for data preparation
from sklearn.model_selection import train_test_split
#training models
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
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
if ('Timestamp') in data:
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

#Covariance matrix and variability comparison between catagories of variables
cmatrix = data.corr()
cf, cax = plt.subplots()
sns.heatmap(cmatrix, cmap='Purples' , annot=True, fmt='.2f')
plt.show()

#Splitting the data
independent_vars = ['family_size', 'annual_income', 'eating_habits', 'addiction_friend', 'addiction', 'medical_history', 'depressed', 'anxiety', 'happy_currently']
X = data[independent_vars] 
y = data['suicidal_thoughts']

#Splitting X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
#Dictionary to store accuracy results of different algorithms
accuracyDict = {}

#Acertaining the feature importance
frst = ExtraTreesClassifier(random_state = 0)
frst.fit(X,y)
imp = frst.feature_importances_
stan_dev = np.std([tree.feature_importances_ for tree in frst.estimators_], axis = 0)

indices = np.argsort(imp)[::-1]
labels = []
for f in range(X.shape[1]):
    labels.append(independent_vars[f])

plt.figure(figsize=(12,8))
plt.title("Feature importances")
plt.bar(range(X.shape[1]), imp[indices],
       color="g", yerr=stan_dev[indices], align="center")
plt.xticks(range(X.shape[1]), labels, rotation='vertical')
plt.xlim([-1, X.shape[1]])
plt.show()

