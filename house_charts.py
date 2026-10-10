import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import math

import pandas as pd
import matplotlib

matplotlib.use("TkAgg")

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# ==========================================
# 1. LOAD DATA AND TRAIN THE MACHINE LEARNING MODEL
# ==========================================

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

    required = FEATURES + ["Price_Lakhs"]

    if not set(required).issubset(data.columns):
        raise ValueError(
            "The CSV must contain: " + ", ".join(required)
        )

    data = data.dropna(subset=required).copy()
    data["Location"] = data["Location"].astype(str).str.strip()

    if len(data) < 5:
        raise ValueError("Please provide at least 5 complete rows.")

    if (data["Price_Lakhs"] <= 0).any():
        raise ValueError("House prices must be greater than zero.")

    X = data[FEATURES]
    y = data["Price_Lakhs"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    def create_model():
        preprocessing = ColumnTransformer(
            transformers=[
                (
                    "location",
                    OneHotEncoder(handle_unknown="ignore"),
                    ["Location"],
                )
            ],
            remainder="passthrough",
        )

        return Pipeline([
            ("preprocessing", preprocessing),
            ("regression", LinearRegression()),
        ])

    # Test model: used for honest evaluation charts
    evaluation_model = create_model()
    evaluation_model.fit(X_train, y_train)

    actual_prices = y_test
    predicted_prices = evaluation_model.predict(X_test)

    mae = mean_absolute_error(actual_prices, predicted_prices)
    r2 = r2_score(actual_prices, predicted_prices)

    # Final model: trained on all rows for user predictions
    model = create_model()
    model.fit(X, y)

    locations = sorted(data["Location"].unique().tolist())

    if not locations:
        raise ValueError("No locations were found in the dataset.")

except Exception as error:
    root = tk.Tk()
    root.withdraw()

    messagebox.showerror(
        "Dataset or Model Error",
        f"Could not load the dataset or train the model.\n\n{error}\n\n"
        "Check that house_data.csv is in the same folder as this file.",
    )

    root.destroy()
    raise SystemExit(1)


# ==========================================
# 2. CREATE THE TKINTER WINDOW
# ==========================================

root = tk.Tk()
root.title("HomeValue - House Price Prediction")
root.geometry("1000x760")
root.minsize(760, 600)
root.configure(bg="#F3F6F8")

DARK = "#173B35"
GREEN = "#23775D"
BG = "#F3F6F8"
WHITE = "#FFFFFF"
TEXT = "#263238"

style = ttk.Style()
style.theme_use("clam")

style.configure(
    "TNotebook",
    background=BG,
    borderwidth=0,
)

style.configure(
    "TNotebook.Tab",
    padding=(18, 10),
    font=("Segoe UI", 10, "bold"),
)

style.map(
    "TNotebook.Tab",
    background=[("selected", WHITE)],
    foreground=[("selected", GREEN)],
)

header = tk.Frame(root, bg=DARK, padx=24, pady=18)
header.pack(fill="x")

tk.Label(
    header,
    text="HomeValue",
    font=("Segoe UI", 24, "bold"),
    bg=DARK,
    fg=WHITE,
).pack(anchor="w")

tk.Label(
    header,
    text="HOUSE PRICE PREDICTION & ANALYTICS",
    font=("Segoe UI", 10),
    bg=DARK,
    fg="#D1E8DC",
).pack(anchor="w", pady=(4, 0))

notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True, padx=18, pady=18)

prediction_tab = tk.Frame(notebook, bg=WHITE)
location_tab = tk.Frame(notebook, bg=WHITE)
performance_tab = tk.Frame(notebook, bg=WHITE)

notebook.add(prediction_tab, text="Predict Price")
notebook.add(location_tab, text="Location Chart")
notebook.add(performance_tab, text="Model Performance")


# ==========================================
# 3. PREDICTION FORM
# ==========================================

tk.Label(
    prediction_tab,
    text="Enter House Details",
    font=("Segoe UI", 20, "bold"),
    bg=WHITE,
    fg=TEXT,
).pack(anchor="w", padx=30, pady=(25, 15))

form = tk.Frame(prediction_tab, bg=WHITE)
form.pack(fill="x", padx=25)

area_var = tk.StringVar(value="1500")
bedrooms_var = tk.StringVar(value="3")
bathrooms_var = tk.StringVar(value="2")
age_var = tk.StringVar(value="5")
location_var = tk.StringVar(value=locations[0])

variables = [
    ("Area (square feet)", area_var),
    ("Bedrooms", bedrooms_var),
    ("Bathrooms", bathrooms_var),
    ("House age (years)", age_var),
]

entries = {}

for index, (label_text, variable) in enumerate(variables):
    row = index // 2
    column = (index % 2) * 2

    tk.Label(
        form,
        text=label_text,
        font=("Segoe UI", 11, "bold"),
        bg=WHITE,
        fg=TEXT,
    ).grid(
        row=row,
        column=column,
        sticky="w",
        padx=8,
        pady=(8, 4),
    )

    entry = ttk.Entry(
        form,
        textvariable=variable,
        font=("Segoe UI", 12),
        width=24,
    )

    entry.grid(
        row=row,
        column=column + 1,
        sticky="ew",
        padx=8,
        pady=(8, 4),
        ipady=5,
    )

    entries[label_text] = entry

tk.Label(
    form,
    text="Location",
    font=("Segoe UI", 11, "bold"),
    bg=WHITE,
    fg=TEXT,
).grid(row=2, column=0, sticky="w", padx=8, pady=12)

location_box = ttk.Combobox(
    form,
    textvariable=location_var,
    values=locations,
    state="readonly",
    font=("Segoe UI", 12),
)

location_box.grid(
    row=2,
    column=1,
    columnspan=3,
    sticky="ew",
    padx=8,
    pady=12,
    ipady=5,
)

for column in range(4):
    form.columnconfigure(column, weight=1)

result_box = tk.Frame(
    prediction_tab,
    bg="#EAF4EF",
    padx=22,
    pady=18,
)
result_box.pack(fill="x", padx=32, pady=20)

tk.Label(
    result_box,
    text="ESTIMATED HOUSE PRICE",
    font=("Segoe UI", 11, "bold"),
    bg="#EAF4EF",
    fg=GREEN,
).pack(anchor="w")

price_label = tk.Label(
    result_box,
    text="₹ — Lakhs",
    font=("Segoe UI", 27, "bold"),
    bg="#EAF4EF",
    fg=DARK,
)
price_label.pack(anchor="w", pady=5)

rupees_label = tk.Label(
    result_box,
    text="Enter details and click Predict Price.",
    font=("Segoe UI", 11),
    bg="#EAF4EF",
    fg=TEXT,
)
rupees_label.pack(anchor="w")


def predict_price():
    try:
        area = float(area_var.get())
        bedrooms = int(bedrooms_var.get())
        bathrooms = int(bathrooms_var.get())
        age = float(age_var.get())
        location = location_var.get().strip()

        if not all(math.isfinite(v) for v in [area, age]):
            raise ValueError("Area and age must be finite numbers.")

        if area <= 0:
            raise ValueError("Area must be greater than zero.")

        if bedrooms < 1 or bathrooms < 1:
            raise ValueError("Bedrooms and bathrooms must be at least 1.")

        if age < 0:
            raise ValueError("House age cannot be negative.")

        if not location:
            raise ValueError("Please select a location.")

        new_house = pd.DataFrame([{
            "Area_sqft": area,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "House_Age": age,
            "Location": location,
        }])

        price = float(model.predict(new_house)[0])

        price_label.config(text=f"₹ {price:,.2f} Lakhs")
        rupees_label.config(
            text=f"Approximately ₹ {price * 100000:,.0f}"
        )

    except ValueError as error:
        messagebox.showerror("Invalid Input", str(error))

    except Exception as error:
        messagebox.showerror(
            "Prediction Error",
            f"Could not predict the price:\n{error}",
        )


def reset_form():
    area_var.set("")
    bedrooms_var.set("")
    bathrooms_var.set("")
    age_var.set("")

    if locations:
        location_var.set(locations[0])

    price_label.config(text="₹ — Lakhs")
    rupees_label.config(text="Enter details and click Predict Price.")


buttons = tk.Frame(prediction_tab, bg=WHITE)
buttons.pack(fill="x", padx=32)

tk.Button(
    buttons,
    text="Predict House Price",
    command=predict_price,
    font=("Segoe UI", 12, "bold"),
    bg=GREEN,
    fg=WHITE,
    activebackground=DARK,
    activeforeground=WHITE,
    relief="flat",
    cursor="hand2",
    padx=18,
    pady=12,
).pack(side="left", padx=(0, 10))

tk.Button(
    buttons,
    text="Reset",
    command=reset_form,
    font=("Segoe UI", 12),
    bg="#E8EDF0",
    fg=TEXT,
    relief="flat",
    cursor="hand2",
    padx=22,
    pady=12,
).pack(side="left")


# ==========================================
# 4. MATPLOTLIB CHART HELPER
# ==========================================

def add_chart(parent, figure):
    canvas = FigureCanvasTkAgg(figure, master=parent)
    canvas.draw()

    canvas.get_tk_widget().pack(
        fill="both",
        expand=True,
        padx=12,
        pady=12,
    )

    return canvas


# ==========================================
# 5. BAR CHART: AVERAGE PRICE BY LOCATION
# ==========================================

tk.Label(
    location_tab,
    text="Average House Price by Location",
    font=("Segoe UI", 18, "bold"),
    bg=WHITE,
    fg=TEXT,
).pack(anchor="w", padx=20, pady=(18, 0))

tk.Label(
    location_tab,
    text="Average price from the supplied dataset; values are in lakhs.",
    font=("Segoe UI", 10),
    bg=WHITE,
    fg="#64748B",
).pack(anchor="w", padx=20, pady=(4, 0))

location_averages = (
    data.groupby("Location")["Price_Lakhs"]
    .mean()
    .sort_values(ascending=False)
)

fig1 = Figure(figsize=(8, 5), dpi=100)
ax1 = fig1.add_subplot(111)

bars = ax1.bar(
    location_averages.index,
    location_averages.values,
    color=["#23775D", "#4C9A7A", "#91C8AD"][:len(location_averages)],
)

ax1.set_title("Average Property Price by City", fontsize=14, pad=14)
ax1.set_xlabel("Location")
ax1.set_ylabel("Average price (₹ lakhs)")
ax1.grid(axis="y", linestyle="--", alpha=0.3)
ax1.set_axisbelow(True)

for bar, value in zip(bars, location_averages.values):
    ax1.annotate(
        f"₹{value:.1f} L",
        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        fontsize=10,
    )

fig1.tight_layout()
add_chart(location_tab, fig1)


# ==========================================
# 6. ACTUAL VS PREDICTED SCATTER CHART
# ==========================================

tk.Label(
    performance_tab,
    text="Actual vs Predicted House Prices",
    font=("Segoe UI", 18, "bold"),
    bg=WHITE,
    fg=TEXT,
).pack(anchor="w", padx=20, pady=(18, 0))

tk.Label(
    performance_tab,
    text=(
        f"Test-set MAE: ₹{mae:.2f} lakhs     "
        f"R² score: {r2:.2f}"
    ),
    font=("Segoe UI", 11),
    bg=WHITE,
    fg="#64748B",
).pack(anchor="w", padx=20, pady=(6, 0))

fig2 = Figure(figsize=(8, 5), dpi=100)
ax2 = fig2.add_subplot(111)

actual = actual_prices.to_numpy()
predicted = predicted_prices

ax2.scatter(
    actual,
    predicted,
    color="#23775D",
    edgecolors="white",
    linewidths=0.8,
    s=85,
    alpha=0.9,
    label="Test houses",
)

# Reference line: perfect predictions lie on this line
lower = min(actual.min(), predicted.min())
upper = max(actual.max(), predicted.max())
padding = max((upper - lower) * 0.08, 1)

ax2.plot(
    [lower - padding, upper + padding],
    [lower - padding, upper + padding],
    linestyle="--",
    color="#D97757",
    label="Perfect prediction",
)

ax2.set_title("Actual Prices vs Model Predictions", fontsize=14, pad=14)
ax2.set_xlabel("Actual price (₹ lakhs)")
ax2.set_ylabel("Predicted price (₹ lakhs)")
ax2.legend()
ax2.grid(linestyle="--", alpha=0.3)

fig2.tight_layout()
add_chart(performance_tab, fig2)


# ==========================================
# 7. FOOTER AND START APPLICATION
# ==========================================

tk.Label(
    root,
    text="Educational demonstration • Sample prices are not market valuations",
    font=("Segoe UI", 9),
    bg=BG,
    fg="#64748B",
    pady=8,
).pack(fill="x")

root.mainloop()