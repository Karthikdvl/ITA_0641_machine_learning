import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
file_path = "E:/Machine learning/loan_approval_dataset.csv"
data = pd.read_csv(file_path)

# Removing leading spaces from column names
data.columns = data.columns.str.strip()

# Encoding categorical variables
label_encoder = LabelEncoder()
data['education'] = label_encoder.fit_transform(data['education'])
data['self_employed'] = label_encoder.fit_transform(data['self_employed'])
data['loan_status'] = label_encoder.fit_transform(data['loan_status'])

# Selecting features and target variable
features = data.drop(columns=['loan_id', 'loan_status'])
target = data['loan_status']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Initializing and training the Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
