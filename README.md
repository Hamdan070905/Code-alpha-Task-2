Code-alpha-Task-2  

1: Import Libraries

             > Imports required Python libraries for data handling (pandas, numpy), visualization (matplotlib), preprocessing (LabelEncoder, StandardScaler, SimpleImputer), model building (Logistic Regression, SVM, Random Forest, XGBoost), and evaluation (accuracy_score, roc_curve, etc.).

2: Load Dataset

             > Loads the heart disease dataset from a CSV file into a DataFrame (df).

3: Drop Missing Target Labels

             > Ensures no missing values exist in the target column (Heart Disease Status) by dropping those rows.

4: Encode Categorical Columns

             > Identifies and label-encodes all categorical columns (including the target column) into numeric format so that machine learning models can process them.

5: Impute Missing Values

             > Applies mean imputation to fill in any remaining missing values in all features using SimpleImputer.

6: Define Features & Target

             > Splits the dataset into:
                    > X → input features (independent variables)
                    > y → target output (Heart Disease Status)

7: Scale Features

             > Applies standard scaling (zero mean, unit variance) to X using StandardScaler, which helps improve model performance and convergence.

8: Train-Test Split

             > Splits the dataset into training (80%) and testing (20%) sets for evaluation.

9: Initialize Models

             > Prepares four machine learning classification models:
             > Logistic Regression
             > Support Vector Machine (SVM)
             > Random Forest
             > XGBoost

10: Train & Evaluate

             > Trains each model using the training data.
             > Evaluates each using the test set and prints:
             > Accuracy
             > Classification report (precision, recall, F1-score)

11: Plot ROC Curves

             > Calculates predicted probabilities and plots the ROC curves (Receiver Operating Characteristic) for all models.
             > Displays AUC (Area Under Curve) score, a key indicator of model performance.


