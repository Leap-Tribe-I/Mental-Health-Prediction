# Suicide-Prediction
Leap 4.0 Major Project

MentalGeek Presents Suicide Prediction Program

## Modularized the Program 
# How to run
- for running download both the src folder and suicide.py file in one directory
- next just run it with python command

# data.py file:
- It will clean and run the code

# split.py file:
It will split the data set into training and testing

# feature.py file:
It will determine the important feature

# tuned_algos.py file:
It will implement different prediction algorithms with GridSearch and Randomized Search CV

# eval.py file: 
It will calculate the accuracy of the models

# Output
accuracyDict:

{
 "Log_Reg_mod_tuning": 85.71428571428571,
 "KNN": 90.47619047619048,
 "Decision_Tree": 80.95238095238095,
 "Random_Forest": 85.71428571428571,
 "AdaBoost_rand": 80.95238095238095,
 "Bagging_rand": 85.71428571428571
}
