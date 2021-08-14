# Suicide-Prediction
Leap 4.0 Major Project

<h2>MentalGeeks Presents</h2>
<h3>Suicide Prediction Program</h3>


# Explaination:
- steps ke liye aalag readme banenge
- this is backend detailed file (isme se jitna lage utna use karna , not need to use whole)

===========================================================

## To implement conda environment: (for unix)
to use conda environment (conda package is needed)(to install conda : apt-get install conda)
# use command :
    step 1: create conda environment
        conda env create -f environment.yml
    step 2: activate conda environment
        conda activate SuicidePrediction

===========================================================

## To implement virtual environment:
to use virtual envirnoment (venv is needed)(to install venv : pip install virtualenv)
# use command: (for unix)
    step 1: create envirnoment
        python -m venv suicide_env
    step 2: activate virtualenv
        source suicide_env/bin/activate
    step 3: install dependenies
        pip install -r venvrequire.txt

# use command: (for windows)
    step 1: create envirnoment
        python -m venv suicide_env
    step 2: activate virtualenv
        suicide_env\Scripts\activate.bat
    step 3: install dependenies
        pip install -r venvrequire.txt

===========================================================
## Week-Wise Update:

# Week 0 - Done
System Setup , created Solution Document 
And Go through Requirement (Both Resource And Knowledge)

# Week 1 - Done
Get data gathered, cleaned and encoded

# Week 2 
Defining relationships between data points 
(using covariance Matrix, variability comparison).

# Week 3 
Evaluation of models (Logistic Regression, Kneighbors Classifier)

# Week 4 
Implementation of remaining models 
(Decision Tree Classifier, Random forests, Bagging, Boosting, etc.) 

# Week 5 
Predicting with neural Network and testing 

# Week 6 
Presenting the Minimal Viable Product

==================================================================
## Backend Steps:

1. Data Loading
2. Getting the information abou the Data(like number of rows and columns)

	Data Cleaning

3. clear NaN
4. Again see data info
	N Also check no data is lost
5. remove all un=necessary Data (like timestamp,comments)

	Data Sorting

6. Encode Data in form of index numbers.

7. Get Data average n percentages n etc
	
	Total Number of: 
	Average Numbers of: Males, Females, income
	sort: living condition

8. Plot correaltion matrix.

9. Data Training.

10.Tuning evaluation of matrix.

11. Creating Final Model.

12. Neural Networking with Tensorflow.

========================================================================

- Data loading,checking,cleaning nd encoding
- Correlation matrix
- Data Splitting
- Data feature importance
- Model Evaluation
- Tuning
	With GridsearchCV
	With RandomizedSearchCV
- DnnClassifier
- output

======================================================
## API description:

* / 				-> home page
* /download		-> runs backend
* /upload			-> receive from upload and send to /download
* /download_file	-> gives output in zip

======================================================
## Backend file description:

# DataProcessing
	-> this contains data loading, checking, cleaning n encoding
# CorrelationMatrix
	-> this contains correlation Matrix
# DataSplitting
	-> this contains data spliting
# FeatureImportance 
	-> this contains data feautre importance
# ModelEvaluation
	-> this contains evaulation of model
# TuningWithGridSearchCV
	-> this contains all GridSearchCV tuning
# TuningwithRandomSearchCV
	-> this contains all RandomizedSearchCV tuning
# AccuracyBarGraph
	-> this contains Accuracy Dict Graph plotting
# Modelling
	-> this contains data modelling
# DnnClassifier
	-> this contains neural network part
# Output
	-> this contains code to output final data

==========================================================

Main.py -> runs only backend
App.py  -> it contains api code
