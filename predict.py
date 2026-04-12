import joblib
import pandas as pd

# 🔹 Load trained model + encoders
model = joblib.load("model.pkl")
le_category = joblib.load("le_category.pkl")
le_city = joblib.load("le_city.pkl")
le_condition = joblib.load("le_condition.pkl")
le_hospital = joblib.load("le_hospital.pkl")

# 🔹 Sample input (you can change this)
toy = ["soft toy", 7, "meerut", "good"]

# 🔹 Encode input properly
toy_encoded = [
    le_category.transform([toy[0]])[0],
    toy[1],
    le_city.transform([toy[2]])[0],
    le_condition.transform([toy[3]])[0]
]

# 🔹 Convert to DataFrame (FIX for warning)
toy_df = pd.DataFrame([toy_encoded], columns=["category","age","city","condition"])

# 🔹 Predict
pred = model.predict(toy_df)
hospital = le_hospital.inverse_transform(pred)

# 🔹 (Optional) Confidence score
prob = model.predict_proba(toy_df)
confidence = max(prob[0]) * 100

# 🔹 Output
print("🎯 Best Match:", hospital[0])
print("📊 Confidence:", round(confidence, 2), "%")