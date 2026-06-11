import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


def preprocess_and_create_pipeline():
    numeric_features = ["age", "fever"]
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_features = ["gender", "cough", "city"]
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    clf = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])

    return clf


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.copy()
    df["has_covid"] = df["has_covid"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["has_covid"])
    y = df["has_covid"]
    return X, y


def train_and_save(csv_path, model_path="model.joblib"):
    X, y = load_data(csv_path)
    model = preprocess_and_create_pipeline()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    joblib.dump(model, model_path)
    return {"accuracy": acc}


def load_model(path="model.joblib"):
    return joblib.load(path)


if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else "covid_toy - covid_toy.csv"
    print("Training from:", csv)
    out = train_and_save(csv)
    print("Saved model to model.joblib; accuracy:", out["accuracy"]) 
