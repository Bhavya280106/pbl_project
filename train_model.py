import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Dataset
data = [
["soft toy",7,"meerut","good","AIIMS Delhi"],
["soft toy",6,"delhi","good","AIIMS Delhi"],
["book",8,"delhi","excellent","AIIMS Delhi"],
["book",7,"meerut","good","AIIMS Delhi"],
["puzzle",10,"mumbai","good","Tata Memorial"],
["puzzle",11,"mumbai","excellent","Tata Memorial"],
["art kit",9,"mumbai","good","Tata Memorial"],
["art kit",8,"mumbai","good","Tata Memorial"],
["soft toy",5,"delhi","good","AIIMS Delhi"],
["book",6,"delhi","good","AIIMS Delhi"],
["puzzle",12,"mumbai","excellent","Tata Memorial"],
["art kit",10,"mumbai","good","Tata Memorial"],
["soft toy",7,"meerut","excellent","AIIMS Delhi"],
["book",9,"delhi","good","AIIMS Delhi"],
["puzzle",10,"mumbai","good","Tata Memorial"],
["soft toy",6,"delhi","good","AIIMS Delhi"],
["book",8,"meerut","good","AIIMS Delhi"],
["puzzle",11,"mumbai","excellent","Tata Memorial"],
["art kit",9,"mumbai","good","Tata Memorial"],
["soft toy",7,"delhi","good","AIIMS Delhi"],
["book",6,"delhi","excellent","AIIMS Delhi"],
["puzzle",10,"mumbai","good","Tata Memorial"],
["art kit",8,"mumbai","good","Tata Memorial"],
["soft toy",5,"meerut","good","AIIMS Delhi"],
["book",7,"delhi","good","AIIMS Delhi"],
["puzzle",12,"mumbai","excellent","Tata Memorial"],
["art kit",9,"mumbai","good","Tata Memorial"],
["soft toy",6,"delhi","good","AIIMS Delhi"],
["book",8,"meerut","good","AIIMS Delhi"],
["puzzle",10,"mumbai","good","Tata Memorial"],
["art kit",9,"mumbai","excellent","Tata Memorial"],
["soft toy",7,"delhi","good","AIIMS Delhi"],
["book",6,"delhi","good","AIIMS Delhi"],
["puzzle",11,"mumbai","excellent","Tata Memorial"],
["art kit",10,"mumbai","good","Tata Memorial"],
["soft toy",5,"meerut","good","AIIMS Delhi"],
["book",7,"delhi","excellent","AIIMS Delhi"],
["puzzle",10,"mumbai","good","Tata Memorial"],
["art kit",8,"mumbai","good","Tata Memorial"],
["soft toy",6,"delhi","good","AIIMS Delhi"],
["book",8,"meerut","good","AIIMS Delhi"],
["puzzle",11,"mumbai","excellent","Tata Memorial"],
["art kit",9,"mumbai","good","Tata Memorial"],
["soft toy",7,"delhi","excellent","AIIMS Delhi"],
["book",6,"delhi","good","AIIMS Delhi"],
["puzzle",10,"mumbai","good","Tata Memorial"],
["art kit",8,"mumbai","good","Tata Memorial"],
]  # (paste above data here)
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