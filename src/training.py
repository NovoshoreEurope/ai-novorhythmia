
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import xgboost as xgb
import joblib
import os
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("data/novorhythmia_dataset.csv")

# Generate text embeddings
print("Generating text embeddings...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
X_text = embedder.encode(df["agenda_text"].tolist())

# Structured features
X_num = df[[
    "hour", "capacity", "total_cost", "availability",
    "extra_hours", "previous_load", "collaborates_with_others"
]].copy()

# Combine text and numeric features
X = np.hstack([X_text, X_num.values])
y = df["predicted_demand"].values

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
print("Training XGBoost model...")
model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("MSE:", round(mean_squared_error(y_test, y_pred), 4))
print("R²:", round(r2_score(y_test, y_pred), 4))

# Save model and embedder
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/novorhythmia_xgb_model.pkl")
embedder.save("models/miniLM_model")
