import pandas as pd
from sklearn.decomposition import non_negative_factorization
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import numpy as np

train_data = pd.read_csv("/Users/tirthpatel/Desktop/Assignemnt 3/prediction_challenge_train.csv")
airport_mapping = pd.read_csv("/Users/tirthpatel/Desktop/Assignemnt 3/airport_country_code_mapping.csv")
test_data = pd.read_csv("/Users/tirthpatel/Desktop/Assignemnt 3/prediction_challenge_test.csv")

train_data = train_data.merge(airport_mapping, on="Airport Country Code", how="left")
test_data = test_data.merge(airport_mapping, on="Airport Country Code", how="left")

for df in [train_data, test_data]:
    df["Departure Date"] = pd.to_datetime(df["Departure Date"], errors="coerce")
    df["Departure Day"] = df["Departure Date"].dt.day
    df["Departure Month"] = df["Departure Date"].dt.month
    df["Departure Year"] = df["Departure Date"].dt.year

train_data["Gender"] = train_data["Gender"].map({"Male": 0, "Female": 1})
test_data["Gender"] = test_data["Gender"].map({"Male": 0, "Female": 1})

train_data["Eligible_For_Discount"] = train_data["Eligible_For_Discount"].map({"Yes": 1, "No": 0})

test_data["Age_Based_Discount"] = "No"
test_data["Month_Based_Discount"] = "No"
test_data["Price_Gender_Discount"] = "No"
test_data["Country_Discount"] = "No"

# Discount for young kids (0-4) and seniors (86-90)
test_data.loc[(test_data["Age"] >= 0) & (test_data["Age"] <= 4), "Age_Based_Discount"] = "Yes"
test_data.loc[(test_data["Age"] >= 86) & (test_data["Age"] <= 90), "Age_Based_Discount"] = "Yes"

# Apply Month-Based Discount
discount_months = [12, 1]
test_data.loc[test_data["Departure Date"].dt.month.isin(discount_months), "Month_Based_Discount"] = "Yes"

# Discounts based on ticket price and gender
male_threshold = np.percentile(train_data["Ticket Price"], 20)
female_threshold = np.percentile(train_data["Ticket Price"], 40)

test_data.loc[(test_data["Gender"] == 0) & (test_data["Ticket Price"] < male_threshold), "Price_Gender_Discount"] = "Yes"
test_data.loc[(test_data["Gender"] == 1) & (test_data["Ticket Price"] < female_threshold), "Price_Gender_Discount"] = "Yes"

print(f'{male_threshold}')
print(f'{female_threshold}')

# Passengers from certain countries gets a discount
discount_countries = ["USA", "UK", "Canada"]
test_data.loc[test_data["Airport Country Code"].isin(discount_countries), "Country_Discount"] = "Yes"

# Combine all discount rules into "Discount_Status"
test_data["Discount_Status"] = "No"
test_data.loc[
    (test_data["Age_Based_Discount"] == "Yes") |
    (test_data["Month_Based_Discount"] == "Yes") |
    (test_data["Price_Gender_Discount"] == "Yes") |
    (test_data["Country_Discount"] == "Yes"),
    "Discount_Status"
] = "Yes"

features = ["Gender", "Age", "Ticket Price", "Departure Day", "Departure Month", "Departure Year"]
X = train_data[features]
y = train_data["Eligible_For_Discount"]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=75)

dt_model = DecisionTreeClassifier(max_depth=5000, min_samples_leaf=1, random_state=75)
dt_model.fit(X_train, y_train)

# Validation
val_predictions = dt_model.predict(X_valid)
validation_acc = accuracy_score(y_valid, val_predictions)
validation_acc = validation_acc*100
print(f"Achieved Validation Accuracy: {validation_acc:.2f}%")

unknown_cases = test_data["Discount_Status"] == "No"
num_unknown = unknown_cases.sum()

if num_unknown > 0:
    test_data.loc[unknown_cases, "Discount_Status"] = dt_model.predict(test_data.loc[unknown_cases, features])
test_data["Discount_Status"] = test_data["Discount_Status"].replace({1: "Yes", 0: "No"})

# Predictions to CSV
test_data[["ID", "Discount_Status", "Age_Based_Discount", "Month_Based_Discount", "Price_Gender_Discount", "Country_Discount"]].to_csv("/Users/tirthpatel/Desktop/Assignemnt 3/predictions.csv", index=False)


