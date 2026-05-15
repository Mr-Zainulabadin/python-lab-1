import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
data = pd.read_csv(r'D:\ml flow\titanic.csv')

# Custom function to impute missing values in the 'Embarked' column
def impute_embarked(X):
    X['Embarked'] = X['Embarked'].fillna(X['Embarked'].mode()[0])  # Fill missing values
    return X

# Custom function to create the 'FamilySize' feature
def create_family_size(X):
    X['FamilySize'] = X['SibSp'] + X['Parch'] + 1  # Adding 1 for the individual themselves
    return X

# Custom function to drop specified columns
def drop_columns(X):
    return X.drop(['SibSp', 'Parch'], axis=1)

# Function to create 'FamilySize' and drop 'SibSp' and 'Parch' columns
def family_size(X):
    X = create_family_size(X)
    X = drop_columns(X)
    return X

# Create pipelines for 'Age'
age_pipeline = Pipeline(steps=[
    ('age_imputer', SimpleImputer(strategy='mean')),  # Impute missing 'Age' values
    ('age_scaler', MinMaxScaler())  # Scale 'Age' feature
])

# Create pipelines for 'Fare'
fare_pipeline = Pipeline(steps=[
    ('fare_scaler', MinMaxScaler())  # Scale 'Fare' feature
])

# Create pipelines for 'FamilySize'
family_size_pipeline = Pipeline(steps=[
    ('family_size_creator', FunctionTransformer(family_size)),
    ('family_size_scaler', MinMaxScaler())  # Scale 'FamilySize' feature
])

# Create pipelines for 'Embarked'
embarked_pipeline = Pipeline(steps=[
    ('embarked_imputer', FunctionTransformer(impute_embarked)),  # Impute missing 'Embarked' values
    ('embarked_onehot', OneHotEncoder())  # One-hot encode 'Embarked'
])

# Create a ColumnTransformer to preprocess the data
rf_preprocessor = ColumnTransformer(transformers=[
    ('drop_columns', 'drop', ['PassengerId', 'Name', 'Ticket', 'Cabin']),  # Drop irrelevant columns
    ('age_encoder', age_pipeline, ['Age']),  # Preprocess 'Age'
    ('fare_encoder', fare_pipeline, ['Fare']),  # Preprocess 'Fare'
    ('family_size', family_size_pipeline, ['SibSp', 'Parch']),  # Preprocess 'FamilySize'
    ('embarked_encoder', embarked_pipeline, ['Embarked']),  # Preprocess 'Embarked'
    ('sex_encoder', OneHotEncoder(), ['Sex']),  # One-hot encode 'Sex'
    ('pclass_scaler', MinMaxScaler(), ['Pclass']),  # Scale 'Pclass'
], remainder='passthrough')

# Create a complete pipeline that includes preprocessing and the Random Forest classifier
rf_pipeline = Pipeline(steps=[
    ('rf_preprocessor', rf_preprocessor),  # Data preprocessing steps
    ('rf_classifier', RandomForestClassifier(n_estimators=100, random_state=42))  # Random Forest Classifier
])

# Separate features and target variable
X = data.drop('Survived', axis=1)
y = data['Survived']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline on the training data
rf_pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_pipeline.predict(X_test)

# Evaluate the model performance
rf_accuracy = accuracy_score(y_test, y_pred)
print(f"\nRandom Forest Model Accuracy: {rf_accuracy:.2f}")

# Confusion matrix for evaluating the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import mlflow
import mlflow.sklearn

# Set the tracking URI and experiment name for Random Forest
mlflow.set_tracking_uri(uri="http://localhost:5000")
mlflow.set_experiment("Random Forest Experiment")

# Start a new MLflow run
with mlflow.start_run():

    # Log the hyperparameters
    mlflow.log_param("model","Random Forest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)

    # Log the accuracy metric
    mlflow.log_metric("accuracy", rf_accuracy)

    # Log the Random Forest model (use the rf_pipeline variable)
    mlflow.sklearn.log_model(rf_pipeline, "Random Forest Algorithm")
