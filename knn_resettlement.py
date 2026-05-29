import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
data = pd.read_csv("refugee_profiles.csv")
print(data.columns)

encoder = LabelEncoder()
city_encoder = LabelEncoder()
data['city'] = city_encoder.fit_transform(data['city'])


data['skill'] = encoder.fit_transform(data['skill'])
data['education_level'] = encoder.fit_transform(data['education_level'])
data['city'] = encoder.fit_transform(data['city'])
X = data[['skill', 'education_level', 'experience_years',
          'openings_count', 'demand_level']]
y = data['city']
scaler = MinMaxScaler()
X = X.copy()

scaled_values = scaler.fit_transform(
    X[['experience_years', 'openings_count']]
)

X.loc[:, 'experience_years'] = scaled_values[:, 0].astype(float)
X.loc[:, 'openings_count'] = scaled_values[:, 1].astype(float)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)


sample = [[2, 3, 5, 100, 3]]  # Example input
prediction = knn.predict(sample)

predicted_city = city_encoder.inverse_transform(prediction)
print("Recommended City:", predicted_city[0])
