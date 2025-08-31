import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import os

# Load dataset
df = pd.read_csv("data/Bangalore_House_Data.csv")

# ✅ Make sure dataset has a 'balcony' column (if not, you can create one manually)
if 'balcony' not in df.columns:
    df['balcony'] = np.random.randint(0, 3, size=len(df))  # random balconies (0–2) for demo

# Encode location
dummies = pd.get_dummies(df['location'])
df = pd.concat([df, dummies], axis=1)

# Features and target (🚫 removed 'bath', ✅ added 'balcony')
X = df.drop(['price', 'location', 'bath'], axis=1, errors="ignore")
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

print("✅ Training Accuracy:", model.score(X_train, y_train))
print("✅ Test Accuracy:", model.score(X_test, y_test))

# Ensure model folder exists
os.makedirs("model", exist_ok=True)

# Save model
with open("model/house_price_model.pkl", "wb") as f:
    pickle.dump((model, X.columns), f)

print("🎉 Model saved to model/house_price_model.pkl")

# To run:
# python train_model.py
# streamlit run app.py

# if not working : venv\Scripts\activate