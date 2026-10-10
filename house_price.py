import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# 1. Load the dataset
data = pd.read_csv("house_data.csv")

print("First five rows of the dataset:")
print(data.head())

# 2. Separate input features and target price
X = data[
    [
        "Area_sqft",
        "Bedrooms",
        "Bathrooms",
        "House_Age",
        "Location",
    ]
]

y = data["Price_Lakhs"]

# 3. Convert location text into machine-readable features
preprocessor = ColumnTransformer(
    transformers=[
        (
            "location",
            OneHotEncoder(handle_unknown="ignore"),
            ["Location"],
        )
    ],
    remainder="passthrough",
)

# 4. Create the machine learning model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ]
)

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# 6. Train the model
model.fit(X_train, y_train)

print("\nModel training completed successfully!")

# 7. Evaluate the model
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"\nMean Absolute Error: {mae:.2f} lakhs")
print(f"R-squared score: {r2:.2f}")

# 8. Get details of a house from the user
print("\nEnter the house details below:")

area = float(input("Area in square feet: "))
bedrooms = int(input("Number of bedrooms: "))
bathrooms = int(input("Number of bathrooms: "))
age = float(input("Age of house in years: "))

print("Available locations: Kozhikode, Kochi, Trivandrum")
location = input("Enter location: ").strip().title()

# 9. Predict the house price
new_house = pd.DataFrame(
    [
        {
            "Area_sqft": area,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "House_Age": age,
            "Location": location,
        }
    ]
)

predicted_price = model.predict(new_house)[0]

print("\n------------------------------")
print("HOUSE PRICE PREDICTION")
print("------------------------------")
print(f"Location: {location}")
print(f"Area: {area:.0f} sq. ft.")
print(f"Predicted price: Rs. {predicted_price:.2f} lakhs")
print(f"Predicted price in rupees: Rs. {predicted_price * 100000:,.0f}")
print("------------------------------")