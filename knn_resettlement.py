import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Load data
data = pd.read_csv("refugee_profiles_expanded.csv")

# 2. Encode the Target Variable (City)
city_encoder = LabelEncoder()
y = city_encoder.fit_transform(data['city'])

# One-Hot Encoding for Features
features_to_encode = ['skill', 'education_level']
X_categorical = pd.get_dummies(data[features_to_encode], drop_first=False)

# Isolate numerical columns
X_numerical = data[['experience_years', 'openings_count']].copy()

# Ensure types are float
X_numerical['experience_years'] = X_numerical['experience_years'].astype(float)
X_numerical['openings_count'] = X_numerical['openings_count'].astype(float)

# Scale training numerical columns
scaler = MinMaxScaler()
numerical_cols = ['experience_years', 'openings_count']
X_numerical[numerical_cols] = scaler.fit_transform(X_numerical[numerical_cols])

# Combine into final Feature Matrix
X = pd.concat([X_categorical, X_numerical], axis=1)

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Model Training (Optimized n_neighbors=1 for high-dimensional sparse data)
knn = KNeighborsClassifier(n_neighbors=1, weights='distance')
knn.fit(X_train, y_train)

# Calculate Accuracy
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)


# ---------------- PREDICTION SECTION ----------------

# Standard raw input profile from a user
sample_raw = {
    'skill': 'electrician',
    'education_level': 'diploma',
    'experience_years': 5.0,
    'openings_count': 100.0
}

# Convert user sample dictionary to a standalone DataFrame copy
sample_df = pd.DataFrame([sample_raw])

# Scale the sample numerical values safely directly on the main sample dataframe
sample_df[numerical_cols] = scaler.transform(sample_df[numerical_cols])

# One-hot encode the sample's text features
sample_encoded = pd.get_dummies(sample_df[['skill', 'education_level']])

# Match the sample's columns to the exact layout the model was trained on
sample_encoded = sample_encoded.reindex(
    columns=X_categorical.columns, fill_value=0)

# Recombine into the final structural sample matrix without any slicing warnings
final_sample = pd.concat([sample_encoded, sample_df[numerical_cols]], axis=1)

# Predict
prediction = knn.predict(final_sample)
predicted_city = city_encoder.inverse_transform(prediction)
print("Recommended City:", predicted_city[0])
