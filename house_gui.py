import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# ---------------------------------
# 1. Load the dataset and train model
# ---------------------------------

DATA_FILE = Path(__file__).resolve().parent / "house_data.csv"

FEATURES = [
    "Area_sqft",
    "Bedrooms",
    "Bathrooms",
    "House_Age",
    "Location",
]

try:
    data = pd.read_csv(DATA_FILE)

    required_columns = FEATURES + ["Price_Lakhs"]

    if not set(required_columns).issubset(data.columns):
        raise ValueError(
            "CSV file is missing required columns. "
            "Please check your house_data.csv file."
        )

    data = data.dropna(subset=required_columns).copy()

    if len(data) < 5:
        raise ValueError(
            "The dataset needs at least 5 complete rows."
        )

    X = data[FEATURES]
    y = data["Price_Lakhs"]

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

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    # Evaluate the model on a test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    evaluation_model = Pipeline(
        steps=[
            ("preprocessor", ColumnTransformer(
                transformers=[
                    (
                        "location",
                        OneHotEncoder(handle_unknown="ignore"),
                        ["Location"],
                    )
                ],
                remainder="passthrough",
            )),
            ("regressor", LinearRegression()),
        ]
    )

    evaluation_model.fit(X_train, y_train)
    test_predictions = evaluation_model.predict(X_test)

    mae = mean_absolute_error(y_test, test_predictions)
    r2 = r2_score(y_test, test_predictions)

    # Train the final prediction model on all available data
    model.fit(X, y)

    locations = sorted(
        data["Location"].astype(str).str.strip().unique().tolist()
    )

except Exception as error:
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(
        "Dataset Error",
        f"Could not load or train the model.\n\n{error}\n\n"
        "Check that house_data.csv is in the same folder "
        "as house_gui.py."
    )
    root.destroy()
    raise SystemExit(1)


# ---------------------------------
# 2. Create the application window
# ---------------------------------

root = tk.Tk()
root.title("HomeValue - House Price Prediction")
root.geometry("650x720")
root.minsize(580, 680)
root.configure(bg="#F3F6F8")

# Theme and colors
DARK_GREEN = "#173B35"
GREEN = "#23775D"
LIGHT_BG = "#F3F6F8"
WHITE = "#FFFFFF"
TEXT_COLOR = "#263238"
MUTED = "#64748B"

style = ttk.Style()
style.theme_use("clam")

style.configure(
    "TCombobox",
    padding=8,
    font=("Segoe UI", 11),
)

style.configure(
    "TEntry",
    padding=9,
    font=("Segoe UI", 11),
)

# Header
header = tk.Frame(root, bg=DARK_GREEN, padx=25, pady=20)
header.pack(fill="x")

tk.Label(
    header,
    text="⌂  HomeValue",
    font=("Segoe UI", 23, "bold"),
    bg=DARK_GREEN,
    fg=WHITE,
).pack(anchor="w")

tk.Label(
    header,
    text="HOUSE PRICE PREDICTION SYSTEM",
    font=("Segoe UI", 10),
    bg=DARK_GREEN,
    fg="#D1E8DC",
).pack(anchor="w", pady=(4, 0))


# ---------------------------------
# 3. Input form
# ---------------------------------

main = tk.Frame(root, bg=WHITE, padx=28, pady=22)
main.pack(fill="both", expand=True, padx=24, pady=20)

tk.Label(
    main,
    text="Enter House Details",
    font=("Segoe UI", 17, "bold"),
    bg=WHITE,
    fg=TEXT_COLOR,
).pack(anchor="w", pady=(0, 18))

form = tk.Frame(main, bg=WHITE)
form.pack(fill="x")

form.columnconfigure(0, weight=1)
form.columnconfigure(1, weight=1)

area_var = tk.StringVar()
bedrooms_var = tk.StringVar()
bathrooms_var = tk.StringVar()
age_var = tk.StringVar()
location_var = tk.StringVar(
    value=locations[0] if locations else ""
)


def add_label(parent, text, row, column):
    tk.Label(
        parent,
        text=text,
        font=("Segoe UI", 10, "bold"),
        bg=WHITE,
        fg=TEXT_COLOR,
    ).grid(
        row=row,
        column=column,
        sticky="w",
        padx=7,
        pady=(5, 6),
    )


def add_entry(parent, variable, row, column):
    entry = ttk.Entry(
        parent,
        textvariable=variable,
        font=("Segoe UI", 11),
    )
    entry.grid(
        row=row,
        column=column,
        sticky="ew",
        padx=7,
        pady=(0, 14),
        ipady=3,
    )
    return entry


add_label(form, "Area (square feet)", 0, 0)
add_label(form, "House age (years)", 0, 1)

area_entry = add_entry(form, area_var, 1, 0)
age_entry = add_entry(form, age_var, 1, 1)

add_label(form, "Bedrooms", 2, 0)
add_label(form, "Bathrooms", 2, 1)

bedrooms_entry = add_entry(form, bedrooms_var, 3, 0)
bathrooms_entry = add_entry(form, bathrooms_var, 3, 1)

add_label(form, "Location", 4, 0)

location_box = ttk.Combobox(
    form,
    textvariable=location_var,
    values=locations,
    state="readonly",
    font=("Segoe UI", 11),
)
location_box.grid(
    row=5,
    column=0,
    columnspan=2,
    sticky="ew",
    padx=7,
    pady=(0, 18),
)


# ---------------------------------
# 4. Prediction result
# ---------------------------------

result_frame = tk.Frame(
    main,
    bg="#EAF4EF",
    padx=18,
    pady=16,
)
result_frame.pack(fill="x", pady=(0, 16))

tk.Label(
    result_frame,
    text="ESTIMATED HOUSE PRICE",
    font=("Segoe UI", 10, "bold"),
    bg="#EAF4EF",
    fg=GREEN,
).pack(anchor="w")

price_label = tk.Label(
    result_frame,
    text="₹ — Lakhs",
    font=("Segoe UI", 25, "bold"),
    bg="#EAF4EF",
    fg=DARK_GREEN,
)
price_label.pack(anchor="w", pady=(5, 2))

rupees_label = tk.Label(
    result_frame,
    text="Enter house details and click Predict.",
    font=("Segoe UI", 10),
    bg="#EAF4EF",
    fg=MUTED,
)
rupees_label.pack(anchor="w")

status_label = tk.Label(
    main,
    text=f"Model MAE: {mae:.2f} lakhs | Test R²: {r2:.2f}",
    font=("Segoe UI", 9),
    bg=WHITE,
    fg=MUTED,
)
status_label.pack(anchor="w", pady=(0, 12))


# ---------------------------------
# 5. Predict button functionality
# ---------------------------------

def predict_price():
    try:
        area_text = area_var.get().strip()
        bedrooms_text = bedrooms_var.get().strip()
        bathrooms_text = bathrooms_var.get().strip()
        age_text = age_var.get().strip()
        location = location_var.get().strip()

        if not all([
            area_text,
            bedrooms_text,
            bathrooms_text,
            age_text,
            location,
        ]):
            messagebox.showwarning(
                "Missing Information",
                "Please fill in all house details.",
            )
            return

        area = float(area_text)
        bedrooms = int(bedrooms_text)
        bathrooms = int(bathrooms_text)
        age = float(age_text)

        if area <= 0:
            raise ValueError("Area must be greater than zero.")

        if bedrooms < 1 or bathrooms < 1:
            raise ValueError(
                "Bedrooms and bathrooms must be at least 1."
            )

        if age < 0:
            raise ValueError("House age cannot be negative.")

        if not all(map(
            lambda value: value < float("inf"),
            [area, age],
        )):
            raise ValueError(
                "Enter valid, finite numbers for area and age."
            )

        new_house = pd.DataFrame([{
            "Area_sqft": area,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "House_Age": age,
            "Location": location,
        }])

        predicted_price = float(model.predict(new_house)[0])

        price_label.config(
            text=f"₹ {predicted_price:,.2f} Lakhs"
        )

        rupees_label.config(
            text=f"Approximately ₹ {predicted_price * 100000:,.0f}"
        )

    except ValueError as error:
        messagebox.showerror(
            "Invalid Input",
            f"Please enter valid house details.\n\n{error}",
        )

    except Exception as error:
        messagebox.showerror(
            "Prediction Error",
            f"Unable to predict the price.\n\n{error}",
        )


# ---------------------------------
# 6. Reset button functionality
# ---------------------------------

def reset_form():
    area_var.set("")
    bedrooms_var.set("")
    bathrooms_var.set("")
    age_var.set("")

    if locations:
        location_var.set(locations[0])

    price_label.config(text="₹ — Lakhs")
    rupees_label.config(
        text="Enter house details and click Predict."
    )


button_frame = tk.Frame(main, bg=WHITE)
button_frame.pack(fill="x")

predict_button = tk.Button(
    button_frame,
    text="Predict House Price",
    command=predict_price,
    font=("Segoe UI", 11, "bold"),
    bg=GREEN,
    fg=WHITE,
    activebackground=DARK_GREEN,
    activeforeground=WHITE,
    relief="flat",
    cursor="hand2",
    padx=16,
    pady=12,
)
predict_button.pack(
    side="left",
    fill="x",
    expand=True,
    padx=(0, 8),
)

reset_button = tk.Button(
    button_frame,
    text="Reset",
    command=reset_form,
    font=("Segoe UI", 11, "bold"),
    bg="#E8EDF0",
    fg=TEXT_COLOR,
    activebackground="#D8E0E5",
    relief="flat",
    cursor="hand2",
    padx=18,
    pady=12,
)
reset_button.pack(side="left")

# Footer
tk.Label(
    root,
    text="Educational demo • Predictions depend on the supplied dataset",
    font=("Segoe UI", 9),
    bg=LIGHT_BG,
    fg=MUTED,
    pady=8,
).pack(fill="x")


# Start the GUI
root.mainloop()