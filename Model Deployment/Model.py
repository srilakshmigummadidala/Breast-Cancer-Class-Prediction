import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# load dataset
df = pd.read_csv('breast-cancer_Healthcare.csv')
df.head()

# replace missing values to NA
df.replace("?", pd.NA, inplace=True)

# find null values
df.isnull().sum()

# Drop rows with missing values
df.dropna(inplace=True)

# Verify the changes
df.isnull().sum()

# Convert "Bare Nuclei" column to int64
df['Bare Nuclei'] = df['Bare Nuclei'].astype('int64')
df.drop(columns =['Sample code number '], inplace = True)

# Replace spaces in column names
df.rename(columns=lambda x: x.strip(), inplace=True)

# Replace spaces in column names with underscores
df.rename(columns=lambda x: x.strip().replace(' ', '_'), inplace=True)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Splitting the dataset into training and testing sets
X = df.drop(columns=['Class'])  # Features
y = df['Class']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification report
print(classification_report(y_test, y_pred))

import pickle
pickle.dump(rf_classifier, open("rf_classifier.pkl","wb"))