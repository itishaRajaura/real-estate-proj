import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Function to retrain the model
def retrain_model(data):
    # Prepare the features and target
    features = data.drop(columns=["price_per_unit_area"])  # Replace with actual target column name
    target = data["price_per_unit_area"]  # Replace with actual target column name
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Initialize the model (RandomForest as an example)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    st.write(f"Model retraining complete. Mean Squared Error: {mse:.2f}")
    
    # Save the retrained model
    joblib.dump(model, "best_model.joblib")
    st.success("Model retrained and saved as 'best_model.joblib'")

# Load trained model (if exists)
try:
    model = joblib.load("best_model.joblib")
except FileNotFoundError:
    st.error("Model file not found! Retraining is required.")
    st.stop()

# Streamlit title and description
st.title("🏘️ Real Estate Price Estimator")
st.write("Estimate the **house price per unit area** based on location and features.")

# Sidebar for retraining the model
st.sidebar.header("Retrain Model")
retrain_file = st.sidebar.file_uploader("Upload your dataset (Excel format)", type=["xlsx"])

if retrain_file:
    data = pd.read_excel(retrain_file)
    st.write("Dataset loaded. Columns in the dataset:")
    st.write(data.columns)

    if st.sidebar.button("Retrain Model"):
        retrain_model(data)

# Sidebar input fields for the user to predict
st.sidebar.header("Property Features")
house_age = st.sidebar.slider("House Age (years)", 0.0, 50.0, 10.0)
dist_to_mrt = st.sidebar.slider("Distance to Nearest MRT (meters)", 0.0, 6500.0, 1000.0)
n_convenience = st.sidebar.number_input("Number of Convenience Stores Nearby", 0, 20, 5)
latitude = st.sidebar.number_input("Latitude", 24.90, 25.10, 24.98)
longitude = st.sidebar.number_input("Longitude", 121.40, 121.60, 121.50)
trans_year = st.sidebar.slider("Transaction Year", 2012, 2013, 2013)
trans_month = st.sidebar.slider("Transaction Month", 1, 12, 6)

# Predict button
if st.sidebar.button("Predict"):
    input_data = pd.DataFrame([{
        "house_age": house_age,
        "dist_to_mrt": dist_to_mrt,
        "n_convenience": n_convenience,
        "latitude": latitude,
        "longitude": longitude,
        "trans_year": trans_year,
        "trans_month": trans_month
    }])

    # Make the prediction
    prediction = model.predict(input_data)[0]
    st.subheader("💰 Predicted Price per Unit Area:")
    st.success(f"{prediction:.2f} (currency units)")

    st.markdown("📌 Note: This is an estimate based on historical data from Taipei housing market.")
