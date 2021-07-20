#importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split


#Splitting the data
def datasplit(data):
    independent_vars = ['family_size', 'annual_income', 'eating_habits', 
                        'addiction_friend', 'addiction', 'medical_history', 
                        'depressed', 'anxiety', 'happy_currently']
    X = data[independent_vars] 
    y = data['suicidal_thoughts']

    #Splitting X and y into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    #returning the X, y, X_train, X_test, y_train, y_test
    return X, y, X_train, X_test, y_train, y_test;