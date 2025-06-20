import pandas as pd
import joblib
from typing import List, Any


def make_prediction(model, input_df: pd.DataFrame, debug: bool = False, as_label: bool = False) -> List[Any]:
    if model is None or input_df is None or input_df.empty:
        return ["Invalid input"]
    try:
        expected_features: list[str] = joblib.load("models/model_features.pkl")
        training_features = [
            "age",
            "rating",
            "blood_pressure",
            "cholesterol",
            "symptom_severity",
            "drug_name",
        ]
        num_cols = ["age", "rating", "symptom_severity"]

        X = input_df[training_features].copy()
        for col in num_cols:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")

        X = pd.get_dummies(X, drop_first=True)
        X.columns = (
            X.columns.str.replace(" ", "_").str.replace("[^A-Za-z0-9_]", "", regex=True)
        )
        X = X.reindex(columns=expected_features, fill_value=0).astype("float64")

        preds = model.predict(X)

        if debug:
            probs = model.predict_proba(X)
            print(
                f"Pred → {preds[0]} (Low Risk p={probs[0][0]:.3f}, High Risk p={probs[0][1]:.3f})"
            )

        if as_label:
            return ["High Risk" if int(p) == 1 else "Low Risk" for p in preds]

        return preds.tolist()

    except Exception as e:
        if debug:
            import traceback

            traceback.print_exc()
        return [f"Prediction failed: {e}"]


def run_quick_test():
    model = joblib.load("models/random_forest_model.pkl")

    samples = [
        {
            "age": 75,
            "drug_name": "Aspirin",
            "blood_pressure": "high",
            "cholesterol": "high",
            "rating": 1,
            "symptom_severity": 9,
        },
        {
            "age": 25,
            "drug_name": "Paracetamol",
            "blood_pressure": "normal",
            "cholesterol": "normal",
            "rating": 5,
            "symptom_severity": 2,
        },
    ]

    for idx, sample in enumerate(samples, start=1):
        label = make_prediction(
            model, pd.DataFrame([sample]), debug=True, as_label=True
        )[0]
        print(f"Sample {idx}: {label}")


if __name__ == "__main__":
    run_quick_test()
