"""Generate ready-to-train Titanic dataset CSVs.

Output goes to ../MLProject/titanic_preprocessing/
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "titanic_raw.csv")
OUT = os.path.join(HERE, "..", "MLProject", "titanic_preprocessing")


def main():
    df = pd.read_csv(RAW)
    print(f"Loaded raw data: {df.shape}")

    y = df["Survived"]
    X = df.drop(columns=["Survived"])
    # Drop non-informative / high-cardinality columns
    X = X.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

    # Handle missing values
    X["Age"] = X["Age"].fillna(X["Age"].median())
    X["Fare"] = X["Fare"].fillna(X["Fare"].median())
    X["Embarked"] = X["Embarked"].fillna(X["Embarked"].mode()[0])

    # Encode categorical features
    X["Sex"] = X["Sex"].map({"male": 0, "female": 1})
    X = pd.get_dummies(X, columns=["Embarked"], drop_first=True)

    print(f"Features: {list(X.columns)}")
    print(f"Any NaN remaining: {X.isna().any().any()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    os.makedirs(OUT, exist_ok=True)
    X_train.to_csv(os.path.join(OUT, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(OUT, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(OUT, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(OUT, "y_test.csv"), index=False)
    print(f"Preprocessed data written to {os.path.abspath(OUT)}")
    print(f"X_train={X_train.shape} X_test={X_test.shape}")


if __name__ == "__main__":
    main()
