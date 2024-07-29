# Importing necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Step 1: Read the Mobile price dataframe using the Pandas module
df = pd.read_csv("E:/Machine learning/mobile_prices.csv")

# Step 2: Print the 1st five rows
print("First five rows of the dataset:")
print(df.head())

# Step 3: Basic statistical computations on the data set or distribution of data
print("\nBasic statistical computations:")
print(df.describe())

# Step 4: The columns and their data types
print("\nColumns and their data types:")
print(df.dtypes)

# Step 5: Detects null values in the dataset. If there are any null values, replace them with the mode value
print("\nNull values in the dataset:")
print(df.isnull().sum())

if df.isnull().sum().any():
    for column in df.columns:
        df[column].fillna(df[column].mode()[0], inplace=True)
        print("\nNull values after filling with mode:")
        print(df.isnull().sum())

# Step 6: Explore the data set using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Step 7: Split the data into test and train sets
X = df.drop('price_range', axis=1)
y = df['price_range']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Fit into the model Naive Bayes Classifier
model = GaussianNB()
model.fit(X_train, y_train)

# Step 9: Predict the model
y_pred = model.predict(X_test)

# Step 10: Find the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy of the Naive Bayes Classifier model:")
print(accuracy)
