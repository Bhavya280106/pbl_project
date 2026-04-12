import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Dataset
data = [

# 👶 SOFT TOYS → Happy Kids (comfort)
["soft toy",2,"delhi","good","Happy Kids Care Center"],
["soft toy",3,"mumbai","good","Happy Kids Care Center"],
["soft toy",4,"jaipur","good","Happy Kids Care Center"],
["soft toy",5,"lucknow","good","Happy Kids Care Center"],
["soft toy",2,"chennai","good","Happy Kids Care Center"],
["soft toy",3,"chandigarh","good","Happy Kids Care Center"],
["soft toy",4,"meerut","good","Happy Kids Care Center"],
["soft toy",5,"delhi","good","Happy Kids Care Center"],
["soft toy",3,"jaipur","good","Happy Kids Care Center"],
["soft toy",2,"lucknow","good","Happy Kids Care Center"],

# 📚 BOOKS → Bright Minds (education)
["book",6,"delhi","good","Bright Minds Learning Hub"],
["book",7,"mumbai","excellent","Bright Minds Learning Hub"],
["book",8,"jaipur","good","Bright Minds Learning Hub"],
["book",9,"lucknow","good","Bright Minds Learning Hub"],
["book",6,"chennai","good","Bright Minds Learning Hub"],
["book",7,"chandigarh","good","Bright Minds Learning Hub"],
["book",8,"meerut","good","Bright Minds Learning Hub"],
["book",9,"delhi","good","Bright Minds Learning Hub"],
["book",6,"jaipur","excellent","Bright Minds Learning Hub"],
["book",7,"lucknow","good","Bright Minds Learning Hub"],

# 🧩 PUZZLES → Brain Boost (cognitive)
["puzzle",8,"delhi","good","Brain Boost Therapy Center"],
["puzzle",9,"mumbai","excellent","Brain Boost Therapy Center"],
["puzzle",10,"jaipur","good","Brain Boost Therapy Center"],
["puzzle",11,"lucknow","good","Brain Boost Therapy Center"],
["puzzle",8,"chennai","good","Brain Boost Therapy Center"],
["puzzle",9,"chandigarh","good","Brain Boost Therapy Center"],
["puzzle",10,"meerut","good","Brain Boost Therapy Center"],
["puzzle",11,"delhi","good","Brain Boost Therapy Center"],
["puzzle",9,"jaipur","excellent","Brain Boost Therapy Center"],
["puzzle",10,"lucknow","good","Brain Boost Therapy Center"],

# 🎨 ART → Creative Zone
["art kit",7,"delhi","good","Creative Kids Zone"],
["art kit",8,"mumbai","good","Creative Kids Zone"],
["art kit",9,"jaipur","good","Creative Kids Zone"],
["art kit",10,"lucknow","good","Creative Kids Zone"],
["art kit",7,"chennai","good","Creative Kids Zone"],
["art kit",8,"chandigarh","good","Creative Kids Zone"],
["art kit",9,"meerut","good","Creative Kids Zone"],
["art kit",10,"delhi","good","Creative Kids Zone"],
["art kit",8,"jaipur","good","Creative Kids Zone"],
["art kit",9,"lucknow","good","Creative Kids Zone"],

# 🌟 PREMIUM → Elite Hospital
["soft toy",6,"delhi","excellent","Elite Children Hospital"],
["book",8,"mumbai","excellent","Elite Children Hospital"],
["puzzle",10,"jaipur","excellent","Elite Children Hospital"],
["art kit",9,"lucknow","excellent","Elite Children Hospital"],
["soft toy",7,"chennai","excellent","Elite Children Hospital"],
["book",9,"chandigarh","excellent","Elite Children Hospital"],
["puzzle",11,"meerut","excellent","Elite Children Hospital"],
["art kit",10,"delhi","excellent","Elite Children Hospital"],

# ⚫ MIX CASES (ML force)
["book",5,"delhi","good","General Toy Care Unit"],
["soft toy",4,"mumbai","good","General Toy Care Unit"],
["puzzle",9,"jaipur","good","General Toy Care Unit"],
["art kit",7,"lucknow","good","General Toy Care Unit"],
["book",6,"chennai","good","General Toy Care Unit"],
["soft toy",3,"chandigarh","good","General Toy Care Unit"],
["puzzle",8,"meerut","good","General Toy Care Unit"],

# 🔥 CROSS MIX (VERY IMPORTANT)
["soft toy",4,"delhi","good","Creative Kids Zone"],
["book",7,"jaipur","good","Brain Boost Therapy Center"],
["puzzle",10,"lucknow","good","Bright Minds Learning Hub"],
["art kit",8,"mumbai","good","Happy Kids Care Center"],
["book",6,"chennai","good","Creative Kids Zone"],
["soft toy",5,"meerut","good","Bright Minds Learning Hub"],
["puzzle",9,"chandigarh","good","Creative Kids Zone"],
["art kit",7,"delhi","good","Brain Boost Therapy Center"],

] # (paste above data here)
print("Data sample:", data[:2])
# Create DataFrame
df = pd.DataFrame(data)
df.columns = ["category","age","city","condition","hospital"]

# Encoders
le_category = LabelEncoder()
le_city = LabelEncoder()
le_condition = LabelEncoder()
le_hospital = LabelEncoder()

df["category"] = le_category.fit_transform(df["category"])
df["city"] = le_city.fit_transform(df["city"])
df["condition"] = le_condition.fit_transform(df["condition"])
df["hospital"] = le_hospital.fit_transform(df["hospital"])

# Features & Target
X = df[["category","age","city","condition"]]
y = df["hospital"]

# Model
model = RandomForestClassifier()
model.fit(X,y)

# Save everything
joblib.dump(model,"model.pkl")
joblib.dump(le_category,"le_category.pkl")
joblib.dump(le_city,"le_city.pkl")
joblib.dump(le_condition,"le_condition.pkl")
joblib.dump(le_hospital,"le_hospital.pkl")

print("✅ Model trained & saved")
print(le_city.classes_)