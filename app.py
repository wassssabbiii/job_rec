from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# ---------------- LOAD DATA ----------------
data = pd.read_csv("refugee_profiles.csv")

data.columns = data.columns.str.strip()
data['skill'] = data['skill'].astype(str).str.strip()
data['education_level'] = data['education_level'].astype(str).str.strip()
data['city'] = data['city'].astype(str).str.strip()

# ---------------- ENCODERS ----------------
encoder_skill = LabelEncoder()
encoder_edu = LabelEncoder()
encoder_city = LabelEncoder()

data['skill_enc'] = encoder_skill.fit_transform(data['skill'])
data['edu_enc'] = encoder_edu.fit_transform(data['education_level'])
data['city_enc'] = encoder_city.fit_transform(data['city'])

# ---------------- FEATURES ----------------
X = data[['skill_enc', 'edu_enc', 'experience_years', 'openings_count']].copy()
y = data['city_enc']

# ---------------- SCALING ----------------
scaler = MinMaxScaler()
X[['experience_years', 'openings_count']] = scaler.fit_transform(
    X[['experience_years', 'openings_count']]
)

# ---------------- MODEL ----------------
knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
knn.fit(X, y)

# ---------------- ROUTE ----------------


@app.route("/", methods=["GET", "POST"])
def index():
    city = None
    message = None

    skills = sorted(encoder_skill.classes_)
    educations = sorted(encoder_edu.classes_)

    if request.method == "POST":
        skill = request.form["skill"]
        education = request.form["education"]
        experience = float(request.form["experience"])
        openings = int(request.form["openings"])

        # 🔴 CONDITION: No jobs available
        if openings == 0:
            message = "No job found for the given input."
        else:
            # Encode inputs
            skill_enc = encoder_skill.transform([skill])[0]
            edu_enc = encoder_edu.transform([education])[0]

            # Prepare input for model
            sample = pd.DataFrame(
                [[skill_enc, edu_enc, experience, openings]],
                columns=[
                    'skill_enc',
                    'edu_enc',
                    'experience_years',
                    'openings_count'
                ]
            )

            sample[['experience_years', 'openings_count']] = scaler.transform(
                sample[['experience_years', 'openings_count']]
            )

            # Predict city
            prediction = knn.predict(sample)
            city = encoder_city.inverse_transform(prediction)[0]

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
