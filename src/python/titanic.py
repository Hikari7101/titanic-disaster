import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")

print("=" * 60)
print("Titanic Survival Prediction - Python")
print("=" * 60)

# ---------- 1. Load training data ----------
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
print(f"\n[1] Loaded train.csv: {train.shape[0]} rows, {train.shape[1]} columns")
print(train.head())

# ---------- 2. Explore ----------
print("\n[2] Missing values in train.csv:")
print(train.isnull().sum())
print(f"\nOverall survival rate: {train['Survived'].mean():.3f}")
print("\nSurvival rate by Sex:")
print(train.groupby("Sex")["Survived"].mean())
print("\nSurvival rate by Pclass:")
print(train.groupby("Pclass")["Survived"].mean())

# ---------- 3. Clean / adjust ----------
age_median = train["Age"].median()
fare_median = train["Fare"].median()
embarked_mode = train["Embarked"].mode()[0]


def prepare(df, name):
    df = df.copy()
    print(f"\n[3] Preparing {name}:")

    n = df["Age"].isnull().sum()
    df["Age"] = df["Age"].fillna(age_median)
    print(f"  - Filled {n} missing Age values with train median ({age_median})")

    n = df["Fare"].isnull().sum()
    df["Fare"] = df["Fare"].fillna(fare_median)
    print(f"  - Filled {n} missing Fare values with train median ({fare_median:.2f})")

    n = df["Embarked"].isnull().sum()
    df["Embarked"] = df["Embarked"].fillna(embarked_mode)
    print(f"  - Filled {n} missing Embarked values with train mode ('{embarked_mode}')")

    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    print("  - Encoded Sex: male=0, female=1")

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    print("  - Added FamilySize = SibSp + Parch + 1")

    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    print("  - Added IsAlone (1 if FamilySize == 1)")

    for port in ["C", "Q", "S"]:
        df[f"Embarked_{port}"] = (df["Embarked"] == port).astype(int)
    print("  - One-hot encoded Embarked into Embarked_C, Embarked_Q, Embarked_S")

    df = df.drop(columns=["Cabin", "Name", "Ticket", "Embarked"])
    print("  - Dropped columns: Cabin (mostly missing), Name, Ticket, Embarked")
    return df


train = prepare(train, "train.csv")

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
            "FamilySize", "IsAlone", "Embarked_C", "Embarked_Q", "Embarked_S"]
print(f"\nFeatures used in the model: {features}")

# ---------- 4. Train logistic regression ----------
X_train = train[features]
y_train = train["Survived"]
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("\n[4] Trained logistic regression model")
print("Coefficients:")
for f, c in zip(features, model.coef_[0]):
    print(f"  {f:12s} {c:+.4f}")

# ---------- 5. Training accuracy ----------
train_acc = accuracy_score(y_train, model.predict(X_train))
print(f"\n[5] Accuracy on training set: {train_acc:.4f}")

# ---------- 6. Predict on test set ----------
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
print(f"\n[6] Loaded test.csv: {test.shape[0]} rows, {test.shape[1]} columns")
test = prepare(test, "test.csv")
test["Survived_pred"] = model.predict(test[features])
print(f"\nPredicted survivors in test set: {test['Survived_pred'].sum()} / {len(test)}")

out_path = os.path.join(DATA_DIR, "predictions_python.csv")
test[["PassengerId", "Survived_pred"]].rename(columns={"Survived_pred": "Survived"}).to_csv(out_path, index=False)
print(f"Saved predictions to {out_path}")

# ---------- 7. Test accuracy ----------
# test.csv has no Survived column, so we compare with gender_submission.csv
labels = pd.read_csv(os.path.join(DATA_DIR, "gender_submission.csv"))
merged = test[["PassengerId", "Survived_pred"]].merge(labels, on="PassengerId")
test_acc = accuracy_score(merged["Survived"], merged["Survived_pred"])
print("\n[7] test.csv has no true labels, so predictions are compared with gender_submission.csv")
print(f"Accuracy on test set (vs gender_submission.csv): {test_acc:.4f}")

print("\nDone!")