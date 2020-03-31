# ChangeLocator
ChangeLocator is an ensemble-learning based model to predict whether an off-target site would be edited using DNA sequence based features.

i. To preprocess the CHANGE-seq data, please use the following command:

python data_preprocess.py

ii. To run the model with default parameter configurations, please use the following command:

python ChangeLocator.py

************************************************************************************
## Required pre-installed packages
Phylo-HMRF requires the following packages to be installed:
- Python (tested on version 2.7.15)
- scikit-learn (tested on version 0.18)
- NumPy (tested on version 1.13.3)
- XGBoost (Pleaes following the instructions at https://xgboost.readthedocs.io/en/latest/)
- pandas (tested on version 0.18.1)
