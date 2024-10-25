## Dataset
#I am using the COVID-19 dataset, which contains country-level data on the spread of COVID-19, vaccinations, and government stringency measures.
#**Dataset link: https://docs.owid.io/projects/etl/api/covid/

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# ================================
# Level 1: Data Cleaning
# ================================

data = pd.read_csv("covid_data.csv")

# Fill missing values using forward fill
data.fillna(method='ffill', inplace=True)

# Ensure column names are stripped of leading/trailing spaces and lowercase
data.columns = data.columns.str.strip().str.lower()

# Select relevant columns
relevant_columns = ['country', 'date', 'new_cases', 'total_vaccinations', 'stringency_index']
cleaned_data = data[relevant_columns]

# Group data by 'country' and aggregate metrics
grouped_data = cleaned_data.groupby('country').agg({
    'new_cases': 'sum',  # Sum of new cases per country
    'total_vaccinations': 'max',  # Max vaccinations
    'stringency_index': 'mean'  # Mean stringency index
}).reset_index()

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(grouped_data[['new_cases', 'total_vaccinations', 'stringency_index']])

# Check cleaned and normalized data
print(grouped_data.head())



# ================================
# Level 2: Random Forest Classification
# ================================
# Reference: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

# Ensure column names are cleaned
grouped_data.columns = grouped_data.columns.str.strip().str.lower()

# Create a binary target variable
# If new_cases > median, classify as 1 (High), otherwise 0 (Low)
grouped_data['case_category'] = (grouped_data['new_cases'] > grouped_data['new_cases'].median()).astype(int)

# Check if the 'case_category' column exists
print(grouped_data.head())  # Verify that 'case_category' is created

# Define features and target
X = grouped_data[['total_vaccinations', 'stringency_index']]
y = grouped_data['case_category']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)



# ================================
# Level 3: Model Evaluation
# ================================
# Reference: https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
