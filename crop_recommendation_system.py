

from google.colab import files
uploaded = files.upload()

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# Step 1: Load the dataset
def load_data():
    print("Loading dataset...")
    df = pd.read_csv("Crop_recommendation.csv")  # Make sure this file exists in the same folder
    return df

# Step 2: Train the model
def train_model(df):
    print("Training model...")
    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc * 100:.2f}%")

    # Save model
    joblib.dump(model, "crop_model.pkl")
    print("Model saved to crop_model.pkl")

# Step 3: Predict crop based on input
def predict_crop():
    if not os.path.exists("crop_model.pkl"):
        print("Model not found! Train the model first.")
        return

    model = joblib.load("crop_model.pkl")

    print("\nEnter the following details to get crop recommendation:")
    N = float(input("Nitrogen (N): "))
    P = float(input("Phosphorus (P): "))
    K = float(input("Potassium (K): "))
    temperature = float(input("Temperature (°C): "))
    humidity = float(input("Humidity (%): "))
    ph = float(input("pH: "))
    rainfall = float(input("Rainfall (mm): "))

    features = [[N, P, K, temperature, humidity, ph, rainfall]]
    prediction = model.predict(features)

    print(f"\n✅ Recommended Crop: {prediction[0]}")

# Step 4: Command-line menu
def main():
    print("=== Crop Recommendation System ===")
    while True:
        print("\n1. Train Model")
        print("2. Predict Crop")
        print("3. Exit")
        choice = input("Choose an option: ")

        if choice == "1":
            df = load_data()
            train_model(df)
        elif choice == "2":
            predict_crop()
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()