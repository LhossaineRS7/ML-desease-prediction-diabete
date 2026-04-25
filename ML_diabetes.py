import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, roc_auc_score, confusion_matrix


def load_and_clean_data(file_path):
    """Loads and cleans the diabetes dataset."""
    df = pd.read_csv(file_path)
    cols_to_clean = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    print("Hidden missing values (zeros) before cleaning:")
    print((df[cols_to_clean] == 0).sum())
    df[cols_to_clean] = df[cols_to_clean].replace(0, np.nan)
    imputer = SimpleImputer(strategy='median')
    df[cols_to_clean] = imputer.fit_transform(df[cols_to_clean])
    print("\nMissing values remaining after imputation:")
    print(df.isnull().sum())
    return df


def prepare_data_for_modeling(df):
    """Splits and scales the data for modeling."""
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("\nFirst row of scaled training data (Notice how the values are now normalized):")
    print(X_train_scaled[0])
    return X_train_scaled, X_test_scaled, y_train, y_test


def train_and_evaluate_logistic_regression(X_train, y_train, X_test):
    """Trains and evaluates a logistic regression model."""
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train, y_train)
    lr_predictions = log_reg.predict(X_test)
    lr_probabilities = log_reg.predict_proba(X_test)[:, 1]
    print("Logistic Regression model trained successfully!")
    return lr_predictions, lr_probabilities


def train_and_evaluate_nn(X_train, y_train, X_test):
    """Trains and evaluates a neural network model."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

    nn_model = Sequential([
        Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("Training Neural Network... This might take a few seconds.")
    nn_model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0)
    nn_probabilities = nn_model.predict(X_test).ravel()
    nn_predictions = (nn_probabilities > 0.5).astype(int)
    print("\nNeural Network trained successfully!")
    return nn_predictions, nn_probabilities


def evaluate_model(model_name, y_true, y_pred):
    """Prints evaluation metrics for a model."""
    print(f"=== {model_name} ===")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred):.3f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.3f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}\n")


def plot_roc_curves(y_test, lr_probabilities, nn_probabilities):
    """Plots the ROC curves for the models and saves it to a file."""
    lr_auc = roc_auc_score(y_test, lr_probabilities)
    nn_auc = roc_auc_score(y_test, nn_probabilities)

    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probabilities)
    nn_fpr, nn_tpr, _ = roc_curve(y_test, nn_probabilities)

    plt.figure(figsize=(10, 7))
    plt.plot(lr_fpr, lr_tpr, linewidth=2, label=f'Logistic Regression (AUC = {lr_auc:.3f})')
    plt.plot(nn_fpr, nn_tpr, linewidth=2, label=f'Neural Network (AUC = {nn_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Guessing (AUC = 0.500)')
    plt.title('ROC Curve: Early Diabetes Prediction Models', fontsize=14, fontweight='bold')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.savefig('roc_curve.png')
    print("\nROC curve plot has been saved to roc_curve.png")


def main():
    """Main function to run the script."""
    df = load_and_clean_data('diabetes.csv')
    X_train_scaled, X_test_scaled, y_train, y_test = prepare_data_for_modeling(df)

    lr_predictions, lr_probabilities = train_and_evaluate_logistic_regression(X_train_scaled, y_train, X_test_scaled)
    evaluate_model("Logistic Regression", y_test, lr_predictions)

    nn_predictions, nn_probabilities = train_and_evaluate_nn(X_train_scaled, y_train, X_test_scaled)
    evaluate_model("Neural Network", y_test, nn_predictions)

    plot_roc_curves(y_test, lr_probabilities, nn_probabilities)


if __name__ == "__main__":
    main()
