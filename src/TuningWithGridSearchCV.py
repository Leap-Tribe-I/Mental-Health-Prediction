# importing from src
from src.ModelEvaluation import evalModel

# sklearn module for tuning
from sklearn.model_selection import GridSearchCV

# sklearn modules for model creation
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import StackingClassifier

# Run all model in one shot
def GridSearchCV(X_train, X_test, y_train, y_test, accuracyDict):
    log_reg_mod_tuning(X_train, X_test, y_train, y_test, accuracyDict)
    tuneKNN(X_train, X_test, y_train, y_test, accuracyDict)
    tuneDT(X_train, X_test, y_train, y_test, accuracyDict)
    tuneRF(X_train, X_test, y_train, y_test, accuracyDict)
    tuneBoosting(X_train, X_test, y_train, y_test, accuracyDict)
    tuneBagging(X_train, X_test, y_train, y_test, accuracyDict)
    # tuneStacking(X_train, X_test, y_train, y_test, accuracyDict)

# tuning the logistic regression model with Gridsearchcv
def log_reg_mod_tuning(X_train, X_test, y_train, y_test, accuracyDict):
    print("\nTuning the Logistic Regression Model with GridSearchCV\n")
    param_grid = {'C':[0.1,1,10,100,1000],
                  'solver':['newton-cg','lbfgs','sag'],
                  'multi_class':['ovr','multinomial'],
                  'max_iter':[100,200,300,400,500]}
    log_reg = LogisticRegression()
    grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train,y_train)
    print("Best parameters: ", grid_search.best_params_)
    print("Best cross-validation score: ", grid_search.best_score_*100, "%")
    print("Best estimator: ", grid_search.best_estimator_)
    lr = grid_search.best_estimator_
    y_pred_class = lr.predict(X_test)
    accuracy = evalModel(lr, X_test, y_test, y_pred_class)
    accuracyDict['Log_Reg_mod_tuning_GridSearchCV'] = accuracy * 100

# tuning the KNN model with GridSearchCV
def tuneKNN(X_train, X_test, y_train, y_test, accuracyDict):
    print("\nTuning KNN model with GridSearchCV\n")
    param_grid = {'n_neighbors':[3,5,7,9,11,13,15],
                  'weights':['uniform','distance'],
                  'algorithm':['auto','ball_tree','kd_tree','brute'],
                  'leaf_size':[10,20,30,40,50,60,70,80]}
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid.fit(X_train,y_train)
    print("Best parameters: ", grid.best_params_)
    print("Best cross-validation score: ", grid.best_score_*100, "%")
    print("Best estimator: ", grid.best_estimator_)
    knn = grid.best_estimator_
    y_pred_class = knn.predict(X_test)
    accuracy = evalModel(knn,X_test, y_test, y_pred_class)
    accuracyDict['KNN_tuning_GridSearchCV'] = accuracy * 100

# tuning the Decision Tree model with GridSearchCV
def tuneDT(X_train, X_test, y_train, y_test, accuracyDict):
    print("\nTuning Decision Tree model with GridSearchCV\n")
    param_grid = {'criterion':['gini','entropy'],
                  'max_depth':[3,5,7,9,11,13,15],
                  'min_samples_split':[2,3,4,5,6,7,8],
                  'random_state':[0]}
    grid = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
    grid.fit(X_train,y_train)
    print("Best parameters: ", grid.best_params_)
    print("Best cross-validation score: ", grid.best_score_*100, "%")
    print("Best estimator: ", grid.best_estimator_)
    dt = grid.best_estimator_
    y_pred_class = dt.predict(X_test)
    accuracy = evalModel(dt,X_test, y_test, y_pred_class)
    accuracyDict['Decision_Tree_tuning_GridSearchCV'] = accuracy * 100

# tuning the Random Forest model with GridSearchCV
def tuneRF(X_train, X_test, y_train, y_test, accuracyDict):
    print("\nTuning Random Forest model with GridSearchCV\n")
    param_grid = {'n_estimators':[10,20,30,40,50,60,70,80,90,100],
                  'max_depth':[3,5,7,9,11,13,15],
                  'min_samples_split':[2,3,4,5,6,7,8],
                  'criterion':['gini','entropy'],
                  'random_state':[0]}
    grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
    grid.fit(X_train,y_train)
    print("Best parameters: ", grid.best_params_)
    print("Best cross-validation score: ", grid.best_score_*100, "%")
    print("Best estimator: ", grid.best_estimator_)
    rf = grid.best_estimator_
    y_pred_class = rf.predict(X_test)
    accuracy = evalModel(rf,X_test, y_test, y_pred_class)
    accuracyDict['Random_Forest_tuning_GridSearchCV'] = accuracy * 100

# tuning boosting model with GridSearchCV
def tuneBoosting(X_train, X_test, y_train, y_test, accuracyDict):
    print("\nTuning Boosting model with GridSearchCV\n")
    param_grid = {'n_estimators':[10,20,30,40,50,60,70,80,90,100],
                  'learning_rate':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                  'random_state':[0]}
    grid = GridSearchCV(AdaBoostClassifier(), param_grid, cv=5)
    grid.fit(X_train,y_train)
    print("Best parameters: ", grid.best_params_)
    print("Best cross-validation score: ", grid.best_score_*100, "%")
    print("Best estimator: ", grid.best_estimator_)
    ada = grid.best_estimator_
    y_pred_class = ada.predict(X_test)
    accuracy = evalModel(ada,X_test, y_test, y_pred_class)
    accuracyDict['AdaBoost_tuning_GridSearchCV'] = accuracy * 100

# tuning bagging model with GridSearchCV
def tuneBagging(X_train, X_test, y_train, y_test, accuracyDict):
    print("\nTuning Bagging model with GridSearchCV\n")
    param_grid = {'n_estimators':[10,20,30,40,50,60,70,80,90,100],
                  'max_samples':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                  'bootstrap':[True,False],
                  'bootstrap_features':[True,False],
                  'random_state':[0]}
    grid = GridSearchCV(BaggingClassifier(), param_grid, cv=5)
    grid.fit(X_train,y_train)
    print("Best parameters: ", grid.best_params_)
    print("Best cross-validation score: ", grid.best_score_*100, "%")
    print("Best estimator: ", grid.best_estimator_)
    bag = grid.best_estimator_
    y_pred_class = bag.predict(X_test)
    accuracy = evalModel(bag,X_test, y_test, y_pred_class)
    accuracyDict['Bagging_tuning_GridSearchCV'] = accuracy * 100

# # tuning stacking model with GridSearchCV
# def tuneStacking(X_train, X_test, y_train, y_test, accuracyDict):
#     # rf = RandomForestClassifier(n_estimators=20, random_state=1)
#     # ada = AdaBoostClassifier(n_estimators=20, learning_rate=0.1, random_state=1)
#     # bag = BaggingClassifier(n_estimators=20, max_samples=0.1, random_state=1)
#     # classifiers=[rf,ada,bag]
#     print("\nTuning Stacking model with GridSearchCV\n")
#     param_grid = {'n_estimators':[10,20,30,40,50,60,70,80,90,100],
#                   'max_samples':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
#                   'bootstrap':[True,False],
#                   'bootstrap_features':[True,False],
#                   'random_state':[0]}
#     grid = GridSearchCV(StackingClassifier(), param_grid, cv=5)
#     grid.fit(X_train,y_train)
#     print("Best parameters: ", grid.best_params_)
#     print("Best cross-validation score: ", grid.best_score_*100, "%")
#     print("Best estimator: ", grid.best_estimator_)
#     stack = grid.best_estimator_
#     y_pred_class = stack.predict(X_test)
#     accuracy = evalModel(stack,X_test,y_test, y_pred_class), y_test, y_pred_class)
#     accuracyDict['Stacking_tuning_GridSearchCV'] = accuracy * 100