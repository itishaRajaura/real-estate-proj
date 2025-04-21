import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Load dataset from Excel
data = pd.read_excel("C:/Users/ITISHA/Downloads/Real Estate Valuation (3)/Real Estate Valuation/Real_estate _valuation_data_set.xlsx") # 🔁 Replace with your filename

# Optional: Rename columns for consistency
data.columns = [
    "trans_year", "trans_month", "house_age", "dist_to_mrt",
    "n_convenience", "latitude", "longitude", "price_per_unit"
]

# Features and target
X = data.drop("price_per_unit", axis=1)
y = data["price_per_unit"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"✅ Model trained. Test MSE: {mse:.2f}")

# Save model
joblib.dump(model, "best_model.joblib")
print("📦 Model saved as best_model.joblib")
