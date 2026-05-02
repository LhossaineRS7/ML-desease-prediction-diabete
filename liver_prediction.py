import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def load_and_clean_data(filepath: str = 'indian_liver_patient.csv') -> Tuple[pd.DataFrame, pd.Series]:
    """Loads and performs initial cleaning on the liver patient data."""
    print(f"Loading and cleaning data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Fix the target column
    df['Dataset'] = df['Dataset'].replace(2, 0)
    
    # Convert 'Gender' to numeric
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

    # Separate features and target
    X = df.drop('Dataset', axis=1)
    y = df['Dataset']
    
    return X, y

def train_and_evaluate_logistic_regression(X_train, y_train, X_test, y_test):
    """Trains and evaluates the Logistic Regression model."""
    print("Training Logistic Regression...")
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)
    
    preds = log_model.predict(X_test)
    probs = log_model.predict_proba(X_test)[:, 1]
    
    print("\n--- Logistic Regression Metrics ---")
    print(f"Accuracy:  {accuracy_score(y_test, preds):.4f}")
    print(f"Precision: {precision_score(y_test, preds):.4f}")
    print(f"Recall:    {recall_score(y_test, preds):.4f}")
    
    return probs
    
def train_and_evaluate_nn(X_train, y_train, X_test, y_test):
    """Trains and evaluates the Neural Network model."""
    print("\nTraining TensorFlow Neural Network...")
    nn_model = Sequential([
        Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train silently (verbose=0) 
    nn_model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=0) 
    
    probs = nn_model.predict(X_test).flatten()
    preds = (probs > 0.5).astype(int)
    
    print("\n--- Neural Network Metrics ---")
    print(f"Accuracy:  {accuracy_score(y_test, preds):.4f}")
    print(f"Precision: {precision_score(y_test, preds):.4f}")
    print(f"Recall:    {recall_score(y_test, preds):.4f}")
    
    return probs
    
def plot_and_save_roc_curve(y_test, log_probs, nn_probs):
    """Calculates AUC and plots the ROC curve for both models."""
    fpr_log, tpr_log, _ = roc_curve(y_test, log_probs)
    roc_auc_log = auc(fpr_log, tpr_log)
    
    fpr_nn, tpr_nn, _ = roc_curve(y_test, nn_probs)
    roc_auc_nn = auc(fpr_nn, tpr_nn)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_log, tpr_log, color='blue', lw=2, label=f'Logistic Regression (AUC = {roc_auc_log:.2f})')
    plt.plot(fpr_nn, tpr_nn, color='red', lw=2, label=f'Neural Network (AUC = {roc_auc_nn:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Liver Disease Prediction', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig('liver_roc_curve.png')
    print("\nROC curve plot has been saved to liver_roc_curve.png")
    plt.show()

def main():
    # 1. Load Data
    X, y = load_and_clean_data()

    # 2. Split Data *before* any preprocessing to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Create a robust preprocessing pipeline using ColumnTransformer and Pipeline
    # This encapsulates imputation and scaling to prevent errors and simplify code.
    impute_col = ['Albumin_and_Globulin_Ratio']
    
    # Create a transformer that imputes the specific column and passes others through
    imputer_transformer = ColumnTransformer(
        transformers=[('imputer', SimpleImputer(strategy='median'), impute_col)],
        remainder='passthrough'  # Keep all other columns
    )
    
    # Create a full pipeline that first imputes, then scales all features
    preprocessing_pipeline = Pipeline([
        ('imputer', imputer_transformer),
        ('scaler', StandardScaler())
    ])

    # 4. Baseline Model
    log_probs = train_and_evaluate_logistic_regression(preprocessing_pipeline.fit_transform(X_train), y_train, preprocessing_pipeline.transform(X_test), y_test)

    # 5. Deep Learning Model
    nn_probs = train_and_evaluate_nn(preprocessing_pipeline.fit_transform(X_train), y_train, preprocessing_pipeline.transform(X_test), y_test)

    # 6. Visualization
    plot_and_save_roc_curve(y_test, log_probs, nn_probs)

if __name__ == "__main__":
    main()
