# =========================================================
# IMPORT LIBRARIES
# =========================================================

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =========================================================
# LOAD DATASET
# =========================================================

data = pd.read_csv(r'D:\ml flow\mllab\labtask\heart_2020_cleaned.csv')

# =========================================================
# FEATURES AND TARGET
# =========================================================

X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']

# Convert target variable into numeric
y = y.map({'Yes': 1, 'No': 0})

# =========================================================
# AUTOMATIC COLUMN DETECTION
# =========================================================

categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(exclude=['object']).columns

print("Categorical Columns:")
print(categorical_cols)

print("\nNumerical Columns:")
print(numerical_cols)

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# NUMERICAL PIPELINE
# =========================================================

numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler())
])

# =========================================================
# CATEGORICAL PIPELINE
# =========================================================

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# =========================================================
# COLUMN TRANSFORMER
# =========================================================

preprocessor = ColumnTransformer(transformers=[

    ('num', numerical_pipeline, numerical_cols),

    ('cat', categorical_pipeline, categorical_cols)

])

# =========================================================
# MODELS
# =========================================================

models = {

    "KNN": KNeighborsClassifier(n_neighbors=5),

    "RandomForest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "DecisionTree": DecisionTreeClassifier(
        criterion='entropy',
        random_state=42
    )
}

# =========================================================
# MLFLOW CONFIGURATION
# =========================================================

mlflow.set_tracking_uri("http://localhost:5000")

mlflow.set_experiment("Heart Disease Prediction")

# =========================================================
# VARIABLES FOR BEST MODEL
# =========================================================

best_accuracy = 0
best_model_name = ""
best_run_id = ""

# =========================================================
# TRAINING LOOP
# =========================================================

for model_name, model in models.items():

    print("\n================================================")
    print(f"TRAINING {model_name}")
    print("================================================")

    # =====================================================
    # CREATE PIPELINE
    # =====================================================

    pipeline = Pipeline(steps=[

        ('preprocessor', preprocessor),

        ('classifier', model)

    ])

    # =====================================================
    # START MLFLOW RUN
    # =====================================================

    with mlflow.start_run(run_name=model_name):

        # =================================================
        # TRAIN MODEL
        # =================================================

        pipeline.fit(X_train, y_train)

        # =================================================
        # PREDICTIONS
        # =================================================

        y_pred = pipeline.predict(X_test)

        # =================================================
        # METRICS
        # =================================================

        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(y_test, y_pred)

        recall = recall_score(y_test, y_pred)

        f1 = f1_score(y_test, y_pred)

        # =================================================
        # PRINT RESULTS
        # =================================================

        print(f"\nAccuracy  : {accuracy:.4f}")

        print(f"Precision : {precision:.4f}")

        print(f"Recall    : {recall:.4f}")

        print(f"F1 Score  : {f1:.4f}")

        print("\nConfusion Matrix:")

        print(confusion_matrix(y_test, y_pred))

        # =================================================
        # LOG PARAMETERS
        # =================================================

        mlflow.log_param("model_name", model_name)

        if model_name == "KNN":

            mlflow.log_param("n_neighbors", 5)

        elif model_name == "RandomForest":

            mlflow.log_param("n_estimators", 100)

        elif model_name == "DecisionTree":

            mlflow.log_param("criterion", "entropy")

        # =================================================
        # LOG METRICS
        # =================================================

        mlflow.log_metric("accuracy", accuracy)

        mlflow.log_metric("precision", precision)

        mlflow.log_metric("recall", recall)

        mlflow.log_metric("f1_score", f1)

        # =================================================
        # LOG MODEL
        # =================================================

        mlflow.sklearn.log_model(

            sk_model=pipeline,

            artifact_path="model"

        )

        # =================================================
        # SAVE BEST MODEL INFO
        # =================================================

        if accuracy > best_accuracy:

            best_accuracy = accuracy

            best_model_name = model_name

            best_run_id = mlflow.active_run().info.run_id

# =========================================================
# REGISTER BEST MODEL
# =========================================================

print("\n================================================")
print("REGISTERING BEST MODEL")
print("================================================")

model_uri = f"runs:/{best_run_id}/model"

registered_model = mlflow.register_model(

    model_uri=model_uri,

    name="BestHeartDiseaseModel"

)

print("\nBest Model Registered Successfully!")

print(f"Best Model: {best_model_name}")

print(f"Best Accuracy: {best_accuracy:.4f}")

# =========================================================
# LOAD REGISTERED MODEL
# =========================================================

print("\n================================================")
print("LOADING REGISTERED MODEL")
print("================================================")

loaded_model = mlflow.pyfunc.load_model(

    model_uri="models:/BestHeartDiseaseModel/latest"

)

# =========================================================
# INFERENCE ON TEST SET
# =========================================================

print("\n================================================")
print("INFERENCE USING REGISTERED MODEL")
print("================================================")

predictions = loaded_model.predict(X_test)

print("\nFirst 20 Predictions:")

print(predictions[:20])

# =========================================================
# FINAL EVALUATION
# =========================================================

final_accuracy = accuracy_score(y_test, predictions)

print(f"\nFinal Accuracy: {final_accuracy:.4f}")