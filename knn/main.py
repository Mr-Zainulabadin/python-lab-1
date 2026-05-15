import mlflow
import mlflow.sklearn

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
titanic_data = pd.read_csv(r'D:\ml flow\titanic.csv')

# Custom function to impute missing values in 'Embarked' column
def impute_embarked(X):
    X['Embarked'] = X['Embarked'].fillna(X['Embarked'].mode()[0])  # Fill missing values
    return X

# Custom function to create the 'FamilySize' feature
def create_family_size(X):
    X['FamilySize'] = X['SibSp'] + X['Parch'] + 1  # Add 1 for the individual themselves
    return X

# Custom function to drop columns that are not needed for model training
def drop_columns(X):
    return X.drop(['SibSp', 'Parch'], axis=1)

# Function to create 'FamilySize' and drop 'SibSp' and 'Parch' columns
def family_size(X):
    #print(X)
    X = create_family_size(X)
    #print(X)
    X = drop_columns(X)
    #print(X)
    return X

# Pipeline to preprocess 'Age' column
age_pipeline = Pipeline(steps=[
    ('age_imputer', SimpleImputer(strategy='mean')),  # Impute missing 'Age' values
    ('age_scaler', MinMaxScaler())  # Scale 'Age' feature
])

# Pipeline to preprocess 'Fare' column
fare_pipeline = Pipeline(steps=[
    #('fare_imputer', SimpleImputer(strategy='mean')),  # Optionally impute missing 'Fare'
    ('fare_scaler', MinMaxScaler())  # Scale 'Fare' feature
])

# Pipeline to create and scale the 'FamilySize' feature
family_size_pipeline = Pipeline(steps=[
    ('family_size_creator', FunctionTransformer(family_size)),
    ('family_size_scaler', MinMaxScaler()),  # Scale 'FamilySize'
])

# Pipeline to preprocess 'Embarked' column
embarked_pipeline = Pipeline(steps=[
    ('embarked_imputer', FunctionTransformer(impute_embarked)),  # Impute missing 'Embarked' values
    ('embarked_onehot', OneHotEncoder())  # One-hot encode 'Embarked'
])

# Create a ColumnTransformer to preprocess all relevant features
knn_preprocessor = ColumnTransformer(transformers=[
    ('age_encoder', age_pipeline, ['Age']),  # Preprocess 'Age'
    ('fare_encoder', fare_pipeline, ['Fare']),  # Preprocess 'Fare'
    ('family_size', family_size_pipeline, ['SibSp', 'Parch']),  # Preprocess 'FamilySize'
    ('embarked_encoder', embarked_pipeline, ['Embarked']),  # Preprocess 'Embarked'
    ('sex_encoder', OneHotEncoder(), ['Sex']),  # One-hot encode 'Sex'
    ('pclass_scaler', MinMaxScaler(), ['Pclass']),  # Scale 'Pclass'
], remainder='passthrough')

# Create a complete pipeline with preprocessing and the KNN classifier
knn_pipeline = Pipeline(steps=[
    ('knn_preprocessor', knn_preprocessor),  # Data preprocessing steps
    ('knn_classifier', KNeighborsClassifier(n_neighbors=5))  # KNN Classifier
])

# Separate features and target variable
#X = data.drop('Survived', axis=1)
X = titanic_data.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
y = titanic_data['Survived']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline on the training data
knn_pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn_pipeline.predict(X_test)

# Evaluate the model performance
knn_accuracy = accuracy_score(y_test, y_pred)
print(f"\nKNN Model Accuracy: {knn_accuracy:.2f}")

# Confusion matrix for evaluating the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Set the tracking URI and experiment name
mlflow.set_tracking_uri(uri="http://localhost:5000")
mlflow.set_experiment("KNN Experiment")

# Start a new MLflow run
with mlflow.start_run():

    # Log the prameters related to KNN model
    mlflow.log_param("model","KNN")
    mlflow.log_param("n_neighbors", 5)
    mlflow.log_param("metric", 'euclidean')

    # Log the accuracy metric
    mlflow.log_metric("accuracy", knn_accuracy)

    # Log the KNN model (use the knn_pipeline variable)
    mlflow.sklearn.log_model(knn_pipeline, "KNN Algorithm")
