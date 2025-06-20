import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# Load and prepare data
df = pd.read_csv("models/cleaned_drug_data.csv")

# Features and target
features = ["age", "rating", "blood_pressure", "cholesterol", "symptom_severity", "drug_name"]
target = "side_effect"

print(f"Dataset shape: {df.shape}")
print(f"Target distribution:\n{df[target].value_counts()}")

# IMPORTANT: Only encode features, NOT the target variable
X_raw = df[features].copy()
y = df[target].copy()

# One-hot encoding for features only
X_encoded = pd.get_dummies(X_raw, drop_first=True)

# Clean column names and ensure proper data types
X_encoded.columns = X_encoded.columns.str.replace(' ', '_').str.replace('[^A-Za-z0-9_]', '', regex=True)

# Ensure all columns are numeric
for col in X_encoded.columns:
    if X_encoded[col].dtype == 'object':
        X_encoded[col] = pd.to_numeric(X_encoded[col], errors='coerce')

# Fill any NaN values
X_encoded = X_encoded.fillna(0)

# Convert to proper data type
X_encoded = X_encoded.astype('float64')

print(f"Encoded features shape: {X_encoded.shape}")
print(f"Feature columns: {list(X_encoded.columns)}")

# Split data for validation
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

print("Training model...")
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("\n=== Model Performance ===")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_encoded.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== Top 10 Most Important Features ===")
print(feature_importance.head(10))

# Save model and feature columns
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/random_forest_model.pkl")
joblib.dump(X_encoded.columns.tolist(), "models/model_features.pkl")

# Save the training data for SHAP background (optional)
background_sample = X_encoded.sample(min(100, len(X_encoded)), random_state=42)
joblib.dump(background_sample, "models/shap_background.pkl")

print("\n✅ Model trained and saved successfully!")
print(f"✅ Saved {len(X_encoded.columns)} feature names")
print("✅ Model ready for prediction!")