import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro, skew, kurtosis
from sklearn.preprocessing import StandardScaler
import joblib  # NEW: For saving scaler

# Load the dataset
df = pd.read_csv("/Users/mikemanner/.cache/kagglehub/datasets/uciml/pima-indians-diabetes-database/versions/1/diabetes.csv")

# Print initial few rows (raw data)
print("First 5 rows before any preprocessing:\n")
print(df.head())

pd.set_option('display.max_rows', None)

# Columns where 0 likely means missing
missing_value_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Replace 0s in each of those columns with the column's median (excluding 0s)
for col in missing_value_columns:
    median_value = df[col][df[col] != 0].median()
    df[col] = df[col].replace(0, median_value)

# Check if any 0s remain
rows_with_missing = df[df[missing_value_columns].isin([0]).any(axis=1)]
print(rows_with_missing)
print(f"Total rows with remaining missing values (0s): {rows_with_missing.shape[0]}")

# Function to check distribution
def check_normality(column_name):
    data = df[column_name]
    plt.hist(data, bins=20, edgecolor='black')
    plt.title(f"Histogram of {column_name}")
    plt.xlabel(column_name)
    plt.ylabel("Frequency")
    plt.show()

    print(f"\nStats for {column_name}:")
    print(f"Skewness: {skew(data):.2f}")
    print(f"Kurtosis: {kurtosis(data):.2f}")

    stat, p = shapiro(data)
    print(f"Shapiro-Wilk Test p-value: {p:.4f}")
    if p > 0.05:
        print("→ Likely normal (bell-shaped)\n")
    else:
        print("→ Not normal (not bell-shaped)\n")

# Run normality check
columns_to_check = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in columns_to_check:
    check_normality(col)

# Normalize all features (exclude 'Outcome')
feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                   'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[feature_columns])

# Save the fitted scaler for inference
joblib.dump(scaler, 'standard_scaler.pkl')
print(" Scaler saved as 'standard_scaler.pkl'")

# Assign scaled values back to the DataFrame
df[feature_columns] = scaled_features

# Preview the normalized data
print("\nPreview of normalized data:")
print(df.head())

# Save cleaned and scaled data to CSV
df.to_csv("cleaned_data.csv", index=False)
print(" Cleaned dataset saved as 'cleaned_data.csv'")
