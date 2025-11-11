from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import pickle, os

app = Flask(__name__)
CORS(app)

# Save files directly in backend folder (no nested path issue)
MODEL_PATH = "health_model.pkl"
DATA_PATH = "health_data.csv"

# Step 1: Create large synthetic dataset if not available
if not os.path.exists(DATA_PATH):
    np.random.seed(42)
    n = 6000  # large dataset for better training
    df = pd.DataFrame({
        "age": np.random.randint(18, 80, n),
        "bmi": np.random.uniform(15, 35, n),
        "hba1c": np.random.uniform(4.5, 9.0, n),
        "activity": np.random.uniform(0, 14, n),
        "ldl": np.random.uniform(70, 200, n)
    })

    # Generate a realistic health score
    df["health_score"] = (
        100
        - (df["bmi"] - 22) ** 2
        - (df["hba1c"] - 5.5) * 6
        + df["activity"] * 2
        - (df["ldl"] - 100) * 0.25
        - (df["age"] - 30) * 0.3
    )
    df["health_score"] = df["health_score"].clip(0, 100)

    df.to_csv(DATA_PATH, index=False)
else:
    df = pd.read_csv(DATA_PATH)

# Step 2: Train or load ML model
if not os.path.exists(MODEL_PATH):
    X = df[["age", "bmi", "hba1c", "activity", "ldl"]]
    y = df["health_score"]
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X, y)
    pickle.dump(model, open(MODEL_PATH, "wb"))
else:
    model = pickle.load(open(MODEL_PATH, "rb"))

# Step 3: Clustering (Big Data insight layer)
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(df[["age", "bmi", "hba1c", "activity", "ldl"]])

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array([
        data["age"], data["bmi"], data["hba1c"], data["activity"], data["ldl"]
    ]).reshape(1, -1)

    predicted_score = float(model.predict(features)[0])
    cluster_label = int(kmeans.predict(features)[0])

    cluster_data = df[df["cluster"] == cluster_label].mean()

    # Generate personalized recommendations
    recos = []
    if data["bmi"] > 27:
        recos.append("⚠️ High BMI: Focus on a low-carb, high-protein diet.")
    elif data["bmi"] < 18.5:
        recos.append("⚠️ Underweight: Eat calorie-dense, nutritious food.")
    else:
        recos.append("✅ Balanced BMI: Maintain your current lifestyle.")

    if data["hba1c"] > 6.4:
        recos.append("⚠️ Elevated HbA1c: Limit sugar and refined carbs.")
    elif data["hba1c"] < 4.5:
        recos.append("⚠️ Low HbA1c: Monitor diet and consult a doctor.")
    else:
        recos.append("✅ Normal HbA1c: Excellent glucose control!")

    if data["activity"] < 3:
        recos.append("🏃 Increase physical activity to 5+ hrs/week.")
    else:
        recos.append("💪 Great activity level! Add strength training.")

    if data["ldl"] > 130:
        recos.append("🍳 High LDL: Avoid processed and fried foods.")
    else:
        recos.append("🥗 LDL is healthy — maintain a balanced diet.")

    if predicted_score < 40:
        recos.append("🚨 Very Low Health Score: Consult a doctor soon.")
    elif predicted_score > 80:
        recos.append("🌟 Excellent score! Keep it up.")

    return jsonify({
        "predicted_health_score": round(predicted_score, 2),
        "cluster": cluster_label,
        "cluster_avg": {
            "bmi": round(cluster_data["bmi"], 2),
            "hba1c": round(cluster_data["hba1c"], 2),
            "ldl": round(cluster_data["ldl"], 2),
            "activity": round(cluster_data["activity"], 2)
        },
        "recommendations": recos
    })


if __name__ == "__main__":
    app.run(debug=True)
