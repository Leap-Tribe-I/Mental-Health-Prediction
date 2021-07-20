import pandas as pd
import numpy as np
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error, precision_recall_curve
#Evaluation of models
def evalModel(model, X_test, y_test, y_pred_class):
    #calculating the accuracy score of the passed model
    acc_score = metrics.accuracy_score(y_test, y_pred_class)
    # print("Accuracy: ", acc_score)
    # print("NULL Accuracy: ", y_test.value_counts())
    # print("Percentage of ones: ", y_test.mean())
    # print("Percentage of zeros: ", 1 - y_test.mean())
    #creating a confunsion matrix
    conmat = metrics.confusion_matrix(y_test, y_pred_class)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    sns.heatmap(conmat, annot=True)
    plt.title("Confusion " + str(model))
    plt.xlabel("predicted")
    plt.ylabel("Actual")
    plt.show()
    #returning accuracy score of the passed algorithm
    return acc_score