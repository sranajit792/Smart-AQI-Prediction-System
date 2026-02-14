import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
df = pd.read_csv("data.csv")

# Keep only required columns
columns = ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3", "AQI"]
df = df[columns]

# Remove rows where AQI is missing
df = df.dropna(subset=["AQI"])

# Fill remaining missing values with column mean
df = df.fillna(df.mean())

# Split features and target
X = df[["PM2.5", "PM10", "NO2", "CO", "SO2", "O3"]]
y = df["AQI"]

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("Model trained successfully and saved as model.pkl")
