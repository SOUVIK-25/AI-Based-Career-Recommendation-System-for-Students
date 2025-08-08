import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import pickle
import os  # 🔹 For creating folder

# Step 1: Sample dataset with 13 input features
data = {
    'Gender': [1, 0, 1, 0, 1, 0, 1, 1, 0, 0],  # 1 = Male, 0 = Female
    'PartTimeJob': [1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
    'ExtraCurricular': [0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
    'SelfStudyHours': [10, 5, 6, 4, 7, 5, 8, 6, 4, 5],
    'Math': [9, 6, 8, 5, 7, 5, 9, 6, 5, 6],
    'History': [4, 9, 5, 10, 6, 9, 3, 4, 8, 7],
    'Physics': [8, 4, 9, 5, 6, 5, 9, 8, 5, 6],
    'Chemistry': [7, 5, 9, 6, 7, 6, 9, 8, 5, 7],
    'Biology': [6, 8, 7, 9, 6, 7, 8, 6, 7, 8],
    'English': [7, 9, 6, 10, 6, 8, 7, 6, 8, 7],
    'Geography': [5, 8, 6, 9, 7, 6, 5, 5, 7, 6],
    'TotalScore': [75, 73, 80, 78, 74, 73, 80, 75, 73, 76],
    'AverageScore': [7.5, 7.3, 8.0, 7.8, 7.4, 7.3, 8.0, 7.5, 7.3, 7.6],
    'Career': [
        'Software Engineer', 'Teacher', 'Data Scientist', 'Historian', 'Biotech Engineer',
        'Journalist', 'AI Engineer', 'Statistician', 'Civil Servant', 'Lawyer'
    ]
}

df = pd.DataFrame(data)

# Step 2: Encode the career column
le = LabelEncoder()
df['CareerLabel'] = le.fit_transform(df['Career'])

# Step 3: Train model
X = df.drop(['Career', 'CareerLabel'], axis=1)
y = df['CareerLabel']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = DecisionTreeClassifier()
model.fit(X_scaled, y)

#  Step 4: Create 'model/' folder if it doesn't exist
os.makedirs("model", exist_ok=True)

#  Step 5: Save model, scaler, and label encoder into 'model/' folder
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("model/labels.pkl", "wb") as f:
    pickle.dump(le, f)

print(" Model, scaler, and labels saved successfully in 'model/' folder.")
