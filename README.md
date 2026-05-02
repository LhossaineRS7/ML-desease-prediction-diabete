# Disease Prediction System

This project is a disease prediction system that uses machine learning to predict the likelihood of a patient having a certain disease. Currently, the system is focused on predicting liver disease.

## Project Overview

This project aims to provide a tool for predicting liver disease in patients based on various medical attributes. It utilizes a machine learning model trained on a dataset of patient information. The primary goal is to create a system that can assist healthcare professionals in making faster and more accurate diagnoses.

The project is built using the following technologies:
- **Python:** The core programming language used for the project.
- **Pandas:** Used for data manipulation and analysis.
- **NumPy:** Used for numerical operations.
- **Scikit-learn:** Used for building and evaluating the machine learning model.
- **TensorFlow:** Used for creating the neural network model.
- **Matplotlib:** Used for generating visualizations, such as the ROC curve.

## How it Works

The `liver_prediction.py` script performs the following steps:

1.  **Data Loading and Preprocessing:**
    *   Loads the patient data from the `indian_liver_patient.csv` file.
    *   Cleans the data by converting categorical features (like 'Gender') into numerical format.
    *   Handles any missing values in the dataset using median imputation.
    *   Splits the data into training and testing sets.
    *   Scales the features using `StandardScaler` to ensure all features have a similar range, which is crucial for neural networks.

2.  **Model Training:**
    *   **Baseline Model:** A Logistic Regression model is trained on the scaled training data to serve as a performance baseline.
    *   **Neural Network:** A deep learning model is built using TensorFlow/Keras. It consists of several dense layers with ReLU activation and a final sigmoid activation layer for binary classification. The model is then trained on the same data.

3.  **Evaluation and Visualization:**
    *   Both models are evaluated on the test set, and their performance is measured using metrics like accuracy, precision, and recall.
    *   An ROC (Receiver Operating Characteristic) curve is generated to visually compare the performance of the Logistic Regression model and the Neural Network.
    *   The resulting ROC curve plot is saved as `liver_roc_curve.png`.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/disease-prediction-system.git
   ```
2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```
3. **Activate the virtual environment:**
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```bash
     source venv/bin/activate
     ```
4. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the liver prediction script:**
   ```bash
   python liver_prediction.py
   ```
2. **The script will output the prediction results, and a ROC curve image `liver_roc_curve.png` will be generated.**

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
