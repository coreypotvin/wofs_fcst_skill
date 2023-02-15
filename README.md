# WOFS_FCST_SKILL

Software package developed to predict forecast skill of the NSSL Warn-on-Forecast System (WoFS)  
Corey Potvin (NSSL), 2023 

## Dependencies

Python >= 3.8  
scikit-learn >= 1.0.2  
numpy >= 1.20.2  
pandas >= 1.4  
matplotlib >= 3.5.1  
xgboost >= 1.5.2  
scikit-optimize >= 0.9.0  
scikit-explain >= 0.0.4  
ml_workflow, available at https://github.com/WarnOnForecast/ml_workflow  
OrdinalClassifier, available at https://github.com/leeprevost/OrdinalClassifier  

## Getting Started

1) conda env create -f environment_py38.yml
2) install ml_workflow
3) download OrdinalClassifier/ordinal.py
4) python imports.py: identify any remaining missing modules

## Functions of scripts

imports.py - imports all modules used by this package; no need to execute manually except to test whether all required modules are installed  
params.py - sets all user-defined parameters/settings used by this package; these are imported by defs.py  
defs. py - contains all custom classes/definitions used by this package; these are imported by each of the main scripts below  
model_train_tune_test.py - model training, hyperparameter tuning, evaluation  
model_reduction.py - reduce model feature set using correlation-based thresholds  
explain_per_fold.py - apply explainability methods to each test fold  
explain_combine_folds.py - aggregate explanations across test folds  

~                               
 
