import tkinter as tk
from tkinter import messagebox
import numpy as np
import pickle

# Load model and scaler
with open("optimized_random_forest.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# GUI window
root = tk.Tk()
root.title("Heart Disease Predictor")
root.geometry("500x600")

# Feature labels (match order of training features)
features = [
    "Age", "Sex (1=Male, 0=Female)", "Chest Pain Type (0-3)", "Resting BP", "Cholesterol",
    "Fasting Blood Sugar (1=True, 0=False)", "Rest ECG (0-2)", "Max Heart Rate",
    "Exercise Induced Angina (1=True, 0=False)", "Oldpeak", "Slope (0-2)", "CA (0-4)", "Thal (0-2)"
]

entries = []

# Create input fields
for i, feature in enumerate(features):
    label = tk.Label(root, text=feature, font=("Arial", 10))
    label.pack()
    entry = tk.Entry(root)
    entry.pack()
    entries.append(entry)

# Predict function
def predict():
    try:
        input_data = [float(entry.get()) for entry in entries]
        input_array = np.array([input_data])
        input_scaled = scaler.transform(input_array)
        pred = rf_model.predict(input_scaled)[0]
        proba = rf_model.predict_proba(input_scaled)[0][1]
        
        if pred == 1:
            message = f"🔴 High Risk of Heart Disease\nProbability: {proba:.2f}"
        else:
            message = f"🟢 Low Risk of Heart Disease\nProbability: {proba:.2f}"
        messagebox.showinfo("Prediction Result", message)
    
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input or model error:\n{e}")

# Predict button
predict_btn = tk.Button(root, text="Predict", command=predict, font=("Arial", 12), bg="blue", fg="white")
predict_btn.pack(pady=20)

root.mainloop()
