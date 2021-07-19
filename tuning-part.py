# tuning the logistic regression model with Gridsearchcv
def log_reg_mod_tuning():
    print("\nTuning the Logistic Regression Model with GridSearchCV\n")
    param_grid = {'C':[0.1,1,10,100,1000],
                  'solver':['newton-cg','lbfgs','sag'],
                  'multi_class':['ovr','multinomial'],
                  'max_iter':[100,200,500,1000]}
    log_reg = LogisticRegression()
    grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train,y_train)
    print("Best score: ", grid_search.best_score_)
    print("Best parameters: ", grid_search.best_params_)
    print("Best estimator: ", grid_search.best_estimator_)
log_reg_mod_tuning()

# tuning the KNN model with GridSearchCV
def tuneKNN():
    print("\nTuning KNN model with GridSearchCV\n")
    param_grid = {'n_neighbors':[3,5,7,9,11,13,15]}
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid.fit(X_train,y_train)
    print("Best parameters: ", grid.best_params_)
    print("Best cross-validation score: ", grid.best_score_)
    print("Best estimator: ", grid.best_estimator_)
tuneKNN()

# tuning the Decision Tree model with GridSearchCV
def tuneDTree():
    print("\nTuning Decision Tree model with GridSearchCV\n")
    param_grid = {'criterion':['gini','entropy'],
                  'max_depth':[3,5,7,9,11,13,15],
                  'min_samples_split':[2,3,4,5,6,7,8]}
    grid = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
    grid.fit(X_train,y_train)
    print("Best parameters: ", grid.best_params_)
    print("Best cross-validation score: ", grid.best_score_)
    print("Best estimator: ", grid.best_estimator_)
tuneDTree()

# tuning the Random Forest model with GridSearchCV
def tuneRF():
    print("\nTuning Random Forest model with GridSearchCV\n")
    param_grid = {'n_estimators':[10,20,30,40,50,60,70,80,90,100],
                  'max_depth':[3,5,7,9,11,13,15],
                  'min_samples_split':[2,3,4,5,6,7,8]}
    grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
    grid.fit(X_train,y_train)
    print("Best parameters: ", grid.best_params_)
    print("Best cross-validation score: ", grid.best_score_)
    print("Best estimator: ", grid.best_estimator_)
tuneRF()