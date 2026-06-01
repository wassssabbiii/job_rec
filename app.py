from flask import Flask, render_template, request
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# ---------------- 1. LOAD DATA ----------------
# Read the expanded dataset and strip trailing spaces
data = pd.read_csv("refugee_profiles_expanded.csv")
data.columns = data.columns.str.strip()

# ---------------- 2. TARGET ENCODING ----------------
city_encoder = LabelEncoder()
y = city_encoder.fit_transform(data['city'].str.strip())

# ---------------- 3. FEATURE ENGINEERING ----------------
# Use dtype=int to force 0/1 integers instead of True/False flags
features_to_encode = ['skill', 'education_level']
X_categorical = pd.get_dummies(
    data[features_to_encode], drop_first=False, dtype=int)

# Isolate numeric columns and convert them safely to floats
X_numerical = data[['experience_years', 'openings_count']].copy()
X_numerical['experience_years'] = X_numerical['experience_years'].astype(float)
X_numerical['openings_count'] = X_numerical['openings_count'].astype(float)

# Scale numeric columns smoothly
scaler = MinMaxScaler()
numerical_cols = ['experience_years', 'openings_count']
X_numerical[numerical_cols] = scaler.fit_transform(X_numerical[numerical_cols])

# Combine categorical columns and numerical columns into our final Feature Matrix
X = pd.concat([X_categorical, X_numerical], axis=1)

# FORCE EVERYTHING IN THE MATRIX TO FLOAT (This fixes the scikit-learn container error permanently)
X = X.astype(float)

# ---------------- 4. MODEL TRAINING ----------------
knn = KNeighborsClassifier(n_neighbors=1, weights='distance')
knn.fit(X, y)

# ---------------- 5. FLASK ROUTE ----------------


@app.route("/", methods=["GET", "POST"])
def index():
    city = None
    message = None

    # Dynamically pull options for HTML dropdown filters
    skills = sorted(data['skill'].unique())
    educations = sorted(data['education_level'].unique())

    if request.method == "POST":
        skill = request.form["skill"]
        education = request.form["education"]
        experience = float(request.form["experience"])
        openings = int(request.form["openings"])

        if openings == 0:
            message = "No job found for the given input."
        else:
            # Create a user profile dictionary mimicking the dataset layout
            sample_raw = {
                'skill': skill,
                'education_level': education,
                'experience_years': experience,
                'openings_count': float(openings)
            }
            sample_df = pd.DataFrame([sample_raw])

            # Scale numeric metrics using training configurations
            sample_df[numerical_cols] = scaler.transform(
                sample_df[numerical_cols])

            # One-Hot Encode user selections with integer settings
            sample_encoded = pd.get_dummies(
                sample_df[['skill', 'education_level']], dtype=int)

            # Reindex to build all structural feature columns, defaulting missing tracks to 0
            sample_encoded = sample_encoded.reindex(
                columns=X_categorical.columns, fill_value=0)

            # Merge columns back together to construct sample matrix
            final_sample = pd.concat(
                [sample_encoded, sample_df[numerical_cols]], axis=1)

            # FORCE USER SAMPLE MATRIX TO FLOAT TYPE (Ensures flawless model prediction parsing)
            final_sample = final_sample.astype(float)

            # Predict
            prediction = knn.predict(final_sample)
            city = city_encoder.inverse_transform(prediction)[0]

    return render_template(
        "index.html",
        city=city,
        message=message,
        skills=skills,
        educations=educations
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
