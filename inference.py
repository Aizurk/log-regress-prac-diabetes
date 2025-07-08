import torch
import torch.nn as nn
import joblib  # ✅ For loading the saved StandardScaler
import numpy as np

# Define the same model class
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Recreate the model and load weights
model = LogisticRegressionModel(input_dim=8)
model.load_state_dict(torch.load("model.pth"))
model.eval()

print("✅ Model loaded. Please enter the following 8 features:")

# List of feature names
features = [
    "Pregnancies",
    "Glucose",
    "Blood Pressure",
    "Skin Thickness",
    "Insulin",
    "BMI",
    "Diabetes Pedigree Function",
    "Age"
]

# Collect user input one by one
input_list = []
for feature in features:
    while True:
        try:
            value = float(input(f"{feature}: "))
            input_list.append(value)
            break
        except ValueError:
            print("❌ Please enter a valid number.")

# Load the scaler and scale input
scaler = joblib.load("standard_scaler.pkl")
scaled_input = scaler.transform([input_list])  # Must be 2D array for scaler

# Prepare input tensor
new_tensor = torch.tensor(scaled_input, dtype=torch.float32)

# Run prediction
with torch.no_grad():
    output = model(new_tensor)
    prediction = (output >= 0.5).float()

# Print results
print(f"\n🧪 Probability of diabetes: {output.item():.4f}")
print(f"🔮 Predicted class: {int(prediction.item())} (0 = No, 1 = Yes)")
