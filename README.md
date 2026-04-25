# Early Disease Prediction System 🏥

## Objective
An end-to-end machine learning pipeline designed to predict the risk of diabetes using the Pima Indians dataset. This project demonstrates robust data engineering practices, including missing value imputation, feature scaling, and comparative model evaluation.

## Architecture & Tech Stack
* **Data Engineering:** Pandas, Scikit-learn (SimpleImputer, StandardScaler)
* **Baseline Model:** Logistic Regression
* **Deep Learning Model:** Multi-Layer Perceptron (MLP) built with TensorFlow/Keras
* **Environment:** Python 3, Linux 

## Key Results
* Successfully handled biologically impossible hidden missing values (0s) via median imputation.
* Designed a custom neural network architecture optimized for tabular healthcare data.
* Evaluated models emphasizing **Recall** to minimize dangerous False Negatives in a medical context.