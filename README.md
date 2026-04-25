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

---

## How to Run This Project

Follow these steps to set up and run the project on your local machine.

**1. Clone the Repository**

First, download the project from GitHub. Replace `your-username/your-repo-name` with the actual GitHub repository URL.

```sh
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

**2. Create a Virtual Environment (Recommended)**

It's a good practice to create a separate "virtual environment" for the project. This keeps its libraries isolated from other Python projects.

```sh
# Create the environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

**3. Install the Required Libraries**

Use `pip` to install all the libraries listed in the `requirements.txt` file with a single command:

```sh
pip install -r requirements.txt
```

**4. Run the Script**

Now you can run the machine learning script:

```sh
python ML_diabetes.py
```

You will see the model's performance printed in the terminal, and a file named `roc_curve.png` will be created in the project folder with the resulting graph.
