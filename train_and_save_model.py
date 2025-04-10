
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
url = "https://raw.githubusercontent.com/dphi-official/Datasets/master/autism_data.csv"
df = pd.read_csv(url)

# Drop irrelevant columns
df = df.drop(['contry_of_res', 'age_desc', 'result', 'relation', 'Case_No'], axis=1)
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df = df.dropna()

# Encode categorical features
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Prepare data
X = df.drop('Class/ASD', axis=1)
y = df['Class/ASD']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
