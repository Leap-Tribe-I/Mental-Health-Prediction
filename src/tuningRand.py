import pandas as pd
import numpy as np
from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, mean_squared_error, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import StackingClassifier
import src.modelEvaluator as ev
# sklearn modules for model creation
from sklearn.neighbors import KNeighborsClassifier

# sklearn module for tunning
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV


# tuning the logistic regression model with RandomizedSearchCV
def log_reg_mod_tuning_rand(X_train, X_test, y_train, y_test, accuracyDict):
    print("\nTuning the Logistic Regression Model with RandomizedSearchCV\n")
    param_dist = {"C": sp_randint(1,100),
                  "solver": ["newton-cg", "lbfgs", "sag"],
                  "multi_class": ["ovr", "multinomial"],
                  "max_iter": sp_randint(100,500)}
    log_reg = LogisticRegression()
    rand_search = RandomizedSearchCV(log_reg, param_dist, cv=5, scoring='accuracy')
    rand_search.fit(X_train,y_train)
    print("Best parameters: ", rand_search.best_params_)
    print("Best cross-validation score: ", rand_search.best_score_*100, "%")
    print("Best estimator: ", rand_search.best_estimator_)
    lr = rand_search.best_estimator_
    y_pred_class = lr.predict(X_test)
    accuracy = ev.evalModel(lr,X_test, y_test, y_pred_class)
    accuracyDict['Log_Reg_mod_tuning_rand'] = accuracy * 100

# tuning the KNN model with RandomizedSearchCV
def tuneKNN_rand(X_train, X_test, y_train, y_test, accuracyDict):
    print("\nTuning KNN model with RandomizedSearchCV\n")
    param_dist = {"n_neighbors": sp_randint(1,100),
                  "weights": ["uniform", "distance"],
                  "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                  "leaf_size": sp_randint(10,100)}
    grid = RandomizedSearchCV(KNeighborsClassifier(), param_dist, cv=5)
    grid.fit(X_train,y_train)
    print("Best parameters: ", grid.best_params_)
    print("Best cross-validation score: ", grid.best_score_*100, "%")
    print("Best estimator: ", grid.best_estimator_)
    knn = grid.best_estimator_
    y_pred_class = knn.predict(X_test)
    accuracy = ev.evalModel(knn,X_test, y_test, y_pred_class)
    accuracyDict['KNN_rand'] = accuracy * 100

# tuning the Decision Tree model with RandomizedSearchCV
def tuneDTree_rand(X_train, X_test, y_train, y_test, accuracyDict):
    print("\nTuning Decision Tree model with RandomizedSearchCV\n")
    param_dist = {"criterion": ["gini", "entropy"],
                  "max_depth": sp_randint(1,100),
                  "min_samples_split": sp_randint(2,10),
                  "random_state": [0]}
    grid = RandomizedSearchCV(DecisionTreeClassifier(), param_dist, cv=5)
    grid.fit(X_train,y_train)
    print("Best parameters: ", grid.best_params_)
    print("Best cross-validation score: ", grid.best_score_*100, "%")
    print("Best estimator: ", grid.best_estimator_)
    dt = grid.best_estimator_
    y_pred_class = dt.predict(X_test)
    accuracy = ev.evalModel(dt,X_test, y_test, y_pred_class)
    accuracyDict['Decision_Tree_rand'] = accuracy * 100

# tuning the Random Forest model with RandomizedSearchCV
def tuneRF_rand(X_train, X_test, y_train, y_test, accuracyDict):
    print("\nTuning Random Forest model with RandomizedSearchCV\n")
    param_dist = {"n_estimators": sp_randint(10,100),
                  "max_depth": sp_randint(1,100),
                  "min_samples_split": sp_randint(2,10),
                  "criterion": ["gini", "entropy"],
                  "random_state": [0]}
    grid = RandomizedSearchCV(RandomForestClassifier(), param_dist, cv=5)
    grid.fit(X_train,y_train)
    print("Best parameters: ", grid.best_params_)
    print("Best cross-validation score: ", grid.best_score_*100, "%")
    print("Best estimator: ", grid.best_estimator_)
    rf = grid.best_estimator_
    y_pred_class = rf.predict(X_test)
    accuracy = ev.evalModel(rf,X_test, y_test, y_pred_class)
    accuracyDict['Random_Forest_rand'] = accuracy * 100

# tuning boosting model with RandomizedSearchCV
def boosting_rand(X_train, X_test, y_train, y_test, accuracyDict):
    print("\nTuning Boosting model with RandomizedSearchCV\n")
    param_dist = {"n_estimators": sp_randint(10,100),
                  "learning_rate": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                  "random_state": [0]}
    grid = RandomizedSearchCV(AdaBoostClassifier(), param_dist, cv=5)
    grid.fit(X_train,y_train)
    print("Best parameters: ", grid.best_params_)
    print("Best cross-validation score: ", grid.best_score_*100, "%")
    print("Best estimator: ", grid.best_estimator_)
    ada = grid.best_estimator_
    y_pred_class = ada.predict(X_test)
    accuracy = ev.evalModel(ada, X_test,y_test, y_pred_class)
    accuracyDict['AdaBoost_rand'] = accuracy * 100


# tuning bagging model with RandomizedSearchCV
def bagging_rand(X_train, X_test, y_train, y_test, accuracyDict):
    print("\nTuning Bagging model with RandomizedSearchCV\n")
    param_dist = {"n_estimators": sp_randint(10,100),
                  "max_samples": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                  "bootstrap": [True,False],
                  "bootstrap_features": [True,False],
                  "random_state": [0]}
    grid = RandomizedSearchCV(BaggingClassifier(), param_dist, cv=5)
    grid.fit(X_train,y_train)
    print("Best parameters: ", grid.best_params_)
    print("Best cross-validation score: ", grid.best_score_*100, "%")
    print("Best estimator: ", grid.best_estimator_)
    bag = grid.best_estimator_
    y_pred_class = bag.predict(X_test)
    accuracy = ev.evalModel(bag,X_test, y_test, y_pred_class)
    accuracyDict['Bagging_rand'] = accuracy * 100