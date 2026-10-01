import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------------
# 1. Load dataset
# -----------------------------------

df = pd.read_csv("loan_data.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# -----------------------------------
# 2. Select features
# -----------------------------------

features = [
    "ApplicantIncome",
    "Self_Employed",
    "Credit_History",
    "Property_Area"
]

X = df[features]
y = df["Loan_Status"]


# -----------------------------------
# 3. Convert target Y/N to 1/0
# -----------------------------------

y = y.map({
    "Y": 1,
    "N": 0
})


# -----------------------------------
# 4. Define feature types
# -----------------------------------

numeric_features = [
    "ApplicantIncome",
    "Credit_History"
]

categorical_features = [
    "Self_Employed",
    "Property_Area"
]


# -----------------------------------
# 5. Preprocessing
# -----------------------------------

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_transformer,
        numeric_features
    ),
    (
        "categorical",
        categorical_transformer,
        categorical_features
    )
])


# -----------------------------------
# 6. Create model
# -----------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# -----------------------------------
# 7. Create pipeline
# -----------------------------------

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# -----------------------------------
# 8. Train/test split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------------
# 9. Train model
# -----------------------------------

pipeline.fit(X_train, y_train)


# -----------------------------------
# 10. Evaluate model
# -----------------------------------

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# -----------------------------------
# 11. Save model
# -----------------------------------

joblib.dump(
    pipeline,
    "model_joblib.pkl"
)

print("\nModel saved successfully!")
print("File created: model_joblib.pkl")