from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# load model
model = joblib.load("model.pkl")
le_category = joblib.load("le_category.pkl")
le_city = joblib.load("le_city.pkl")
le_condition = joblib.load("le_condition.pkl")
le_hospital = joblib.load("le_hospital.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        category = data.get("category", "soft toy").lower()
        city = data.get("city", "delhi").lower()
        condition = data.get("condition", "good").lower()
        age = int(data.get("age", 7))

        if "soft" in category:
            category = "soft toy"
        elif "puzzle" in category:
            category = "puzzle"
        elif "art" in category:
            category = "art kit"
        elif "book" in category or "edu" in category:
            category = "book"
        else:
            category = "soft toy"

        if category not in le_category.classes_:
            category = "soft toy"

        if city not in le_city.classes_:
            city = "delhi"

        if condition not in le_condition.classes_:
            condition = "good"

        toy_encoded = [
            le_category.transform([category])[0],
            age,
            le_city.transform([city])[0],
            le_condition.transform([condition])[0]
        ]

        toy_df = pd.DataFrame([toy_encoded], columns=["category","age","city","condition"])

        pred = model.predict(toy_df)
        hospital = le_hospital.inverse_transform(pred)

        return jsonify({"hospital": hospital[0]})

    except Exception as e:
        print("🔥 ERROR:", e)
        return jsonify({"hospital": "AIIMS"})

# 🔥 THIS WAS MISSING
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)