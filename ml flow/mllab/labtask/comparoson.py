import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# =========================================================
# LOAD DATASET
# =========================================================

data = pd.read_csv(r'D:\ml flow\titanic.csv')

# =========================================================
# CUSTOM FUNCTIONS
# =========================================================

def impute_embarked(X):
    X['Embarked'] = X['Embarked'].fillna(X['Embarked'].mode()[0])
    return X

def create_family_size(X):
    X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
    return X

def drop_columns(X):
    return X.drop(['SibSp', 'Parch'], axis=1)

def family_size(X):
    X = create_family_size(X)
    X = drop_columns(X)
    return X

# =========================================================
# PREPROCESSING PIPELINES
# =========================================================

age_pipeline = Pipeline(steps=[
    ('age_imputer', SimpleImputer(strategy='mean')),
    ('age_scaler', MinMaxScaler())
])

fare_pipeline = Pipeline(steps=[
    ('fare_scaler', MinMaxScaler())
])

family_size_pipeline = Pipeline(steps=[
    ('family_size_creator', FunctionTransformer(family_size)),
    ('family_size_scaler', MinMaxScaler())
])

embarked_pipeline = Pipeline(steps=[
    ('embarked_imputer', FunctionTransformer(impute_embarked)),
    ('embarked_onehot', OneHotEncoder())
])

preprocessor = ColumnTransformer(transformers=[
    ('age_encoder', age_pipeline, ['Age']),
    ('fare_encoder', fare_pipeline, ['Fare']),
    ('family_size', family_size_pipeline, ['SibSp', 'Parch']),
    ('embarked_encoder', embarked_pipeline, ['Embarked']),
    ('sex_encoder', OneHotEncoder(), ['Sex']),
    ('pclass_scaler', MinMaxScaler(), ['Pclass']),
], remainder='drop')

# =========================================================
# DATA SPLIT
# =========================================================

X = data.drop(
    ['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'],
    axis=1
)

y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

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
        criterion='entropy'
    )
}

# =========================================================
# MLFLOW SETUP
# =========================================================

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Titanic Model Comparison")

best_accuracy = 0
best_model_name = ""
best_run_id = ""

# =========================================================
# TRAINING LOOP
# =========================================================

for model_name, model in models.items():

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    with mlflow.start_run(run_name=model_name):

        # Train model
        pipeline.fit(X_train, y_train)

        # Predictions
        y_pred = pipeline.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Print results
        print(f"\n===== {model_name} =====")
        print("Accuracy :", accuracy)
        print("Precision:", precision)
        print("Recall   :", recall)
        print("F1 Score :", f1)

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        # =================================================
        # LOG PARAMETERS
        # =================================================

        mlflow.log_param("model", model_name)

        # Extra params
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
        # FIND BEST MODEL
        # =================================================

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = model_name
            best_run_id = mlflow.active_run().info.run_id

# =========================================================
# REGISTER BEST MODEL
# =========================================================

model_uri = f"runs:/{best_run_id}/model"

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name="BestTitanicModel"
)

print("\n===================================")
print("Best Model Registered Successfully")
print("===================================")

print("Best Model :", best_model_name)
print("Best Accuracy :", best_accuracy)

# =========================================================
# LOAD REGISTERED MODEL
# =========================================================

loaded_model = mlflow.pyfunc.load_model(
    model_uri="models:/BestTitanicModel/latest"
)

# =========================================================
# INFERENCE ON TEST SET
# =========================================================

predictions = loaded_model.predict(X_test)

print("\n===================================")
print("Inference Using Registered Model")
print("===================================")

print(predictions[:10])

# Final Accuracy
final_accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy of Registered Model:", final_accuracy)