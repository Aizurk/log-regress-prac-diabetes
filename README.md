# Diabetes Prediction with Logistic Regression

#First fix the file paths to match yours or use enviornment variables. After that you can run inference with the inference.py file.

This project implements a machine learning model to predict diabetes risk using the Pima Indians Diabetes Database.

## Project Structure

- `download_data.py` - Downloads the diabetes dataset from Kaggle
- `inspect_data.py` - Basic data inspection and exploration
- `logistic_regression_model.py` - Complete ML pipeline for training the model
- `predict_diabetes.py` - Script for making predictions on new data
- `requirements.txt` - Required Python packages

## Workflow

### 1. Data Download
```bash
python3 download_data.py
```
This downloads the Pima Indians Diabetes Database from Kaggle.

### 2. Data Inspection
```bash
python3 inspect_data.py
```
This provides basic information about the dataset structure and content.

### 3. Model Training
```bash
python3 logistic_regression_model.py
```
This script:
- Loads and preprocesses the data
- Handles missing values (replaces 0s with NaN, then imputes with median)
- Splits data into training and test sets
- Scales features using StandardScaler
- Trains a logistic regression model
- Evaluates the model performance
- Creates visualizations
- Saves the trained model and scaler

### 4. Making Predictions
```bash
python3 predict_diabetes.py
```
This script allows you to:
- Make predictions on new patient data
- Get probability scores and risk levels
- Receive recommendations based on risk level

## Dataset Features

The dataset contains the following features:
- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration (mg/dL)
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)²)
- **DiabetesPedigreeFunction**: Diabetes pedigree function
- **Age**: Age in years
- **Outcome**: Target variable (0 = no diabetes, 1 = diabetes)

## Model Performance

The logistic regression model typically achieves:
- Accuracy: ~75-80%
- ROC AUC: ~0.75-0.80
- Good balance between precision and recall

## Key Features of the Implementation

1. **Data Preprocessing**:
   - Handles missing values appropriately for medical data
   - Feature scaling for better model performance
   - Stratified sampling to maintain class balance

2. **Model Evaluation**:
   - Multiple evaluation metrics (accuracy, ROC AUC, precision, recall)
   - Confusion matrix visualization
   - Feature importance analysis

3. **Visualizations**:
   - Feature importance plot
   - Target distribution
   - Correlation heatmap
   - Box plots for key features vs outcome

4. **Prediction Interface**:
   - Easy-to-use prediction function
   - Risk level categorization
   - Personalized recommendations

## Usage Example

```python
from predict_diabetes import predict_diabetes_risk

# Example prediction
probability, prediction, risk_level = predict_diabetes_risk(
    glucose=140, blood_pressure=80, skin_thickness=30, insulin=150,
    bmi=28.5, diabetes_pedigree=0.6, age=35, pregnancies=2
)

print(f"Probability: {probability:.3f}")
print(f"Prediction: {'Diabetes' if prediction == 1 else 'No Diabetes'}")
print(f"Risk Level: {risk_level}")
```

## Requirements

Install the required packages:
```bash
pip3 install -r requirements.txt
```

## Notes

- The model is trained on the Pima Indians Diabetes Database, which may not generalize to all populations
- Medical predictions should always be validated by healthcare professionals
- This is for educational/demonstration purposes only

## Output Files

After running the training script, you'll get:
- `diabetes_logistic_model.pkl` - Trained model
- `diabetes_scaler.pkl` - Feature scaler
- `diabetes_analysis.png` - Visualization plots 
