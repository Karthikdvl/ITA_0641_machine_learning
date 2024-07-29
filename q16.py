import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
file_path = 'HousePricePrediction.csv'
data = pd.read_csv(file_path)

# Handling missing values
data = data.dropna(subset=['SalePrice'])  # Drop rows where SalePrice is missing

# Fill missing categorical values with the mode and numerical with the median
for column in data.select_dtypes(include=['object']).columns:
    mode_value = data[column].mode()[0]
    data[column].fillna(mode_value, inplace=True)

for column in data.select_dtypes(include=['float64', 'int64']).columns:
    median_value = data[column].median()
    data[column] = data[column].fillna(median_value)

# Discretize SalePrice into categories (e.g., low, medium, high)
price_bins = [0, 130000, 200000, np.inf]
price_labels = ['low', 'medium', 'high']
data['PriceCategory'] = pd.cut(data['SalePrice'], bins=price_bins, labels=price_labels)

# Encode categorical variables
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Encode target labels as integers
target = LabelEncoder().fit_transform(data['PriceCategory'])

# Selecting features and target variable
features = data.drop(columns=['Id', 'SalePrice', 'PriceCategory'])

# Standardize features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)

# Define classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

# Dictionary to store evaluation metrics
metrics = {
    "Model": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1 Score": []
}

# Train and evaluate each classifier
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    metrics["Model"].append(name)
    metrics["Accuracy"].append(accuracy_score(y_test, y_pred))
    metrics["Precision"].append(precision_score(y_test, y_pred, average='weighted', zero_division=1))
    metrics["Recall"].append(recall_score(y_test, y_pred, average='weighted'))
    metrics["F1 Score"].append(f1_score(y_test, y_pred, average='weighted'))

# Convert metrics to DataFrame for better visualization
metrics_df = pd.DataFrame(metrics)
print(metrics_df)
