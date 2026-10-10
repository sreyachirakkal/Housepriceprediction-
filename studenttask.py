import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

# 1. Create historical student data
data = {
    "study_hours": [
        8, 2, 6, 1, 5, 3, 7, 2, 9, 4,
        8, 3, 6, 1, 7, 4, 9, 2, 5, 6,
        10, 3, 7, 2, 8, 4, 6, 1, 5, 9
    ],
    "attendance": [
        90, 45, 80, 35, 75, 50, 85, 40, 95, 65,
        88, 55, 82, 30, 90, 60, 98, 42, 70, 78,
        96, 48, 87, 38, 92, 62, 81, 28, 72, 94
    ],
    "previous_marks": [
        85, 30, 70, 25, 60, 40, 78, 35, 92, 55,
        80, 42, 75, 20, 85, 50, 95, 28, 65, 72,
        98, 38, 82, 32, 90, 58, 74, 18, 63, 96
    ],
    "result": [
        "Pass", "Fail", "Pass", "Fail", "Pass",
        "Fail", "Pass", "Fail", "Pass", "Pass",
        "Pass", "Fail", "Pass", "Fail", "Pass",
        "Fail", "Pass", "Fail", "Pass", "Pass",
        "Pass", "Fail", "Pass", "Fail", "Pass",
        "Fail", "Pass", "Fail", "Pass", "Pass"
    ]
}

df = pd.DataFrame(data)

print("HISTORICAL STUDENT DATA")
print(df.to_string(index=False))

# 2. Separate features and target
X = df[["study_hours", "attendance", "previous_marks"]]
y = df["result"]

# 3. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# 4. Create and train the machine learning model
model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel trained successfully!")

# 5. Evaluate the model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nActual results:   ", y_test.tolist())
print("Predicted results:", y_pred.tolist())
print(f"Model accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, zero_division=0
))

# 6. Display the confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    labels=["Fail", "Pass"],
    cmap="Blues"
)

plt.title("Student Pass/Fail Confusion Matrix")
plt.tight_layout()
plt.show()

# 7. Predict for a new student
new_student = pd.DataFrame([{
    "study_hours": 6,
    "attendance": 85,
    "previous_marks": 75
}])

prediction = model.predict(new_student)[0]

print("\nNEW STUDENT DETAILS")
print(new_student.to_string(index=False))
print("\nPredicted result:", prediction)

# 8. Visualize the learned decision tree
plt.figure(figsize=(14, 7))

plot_tree(
    model,
    feature_names=X.columns.tolist(),
    class_names=model.classes_.tolist(),
    filled=True,
    rounded=True
)

plt.title("Decision Tree for Student Pass/Fail Prediction")
plt.tight_layout()
plt.show()

# 9. Predict for another student using keyboard input
print("\nENTER DETAILS FOR ANOTHER STUDENT")

try:
    hours = float(input("Study hours per day: "))
    attendance = float(input("Attendance percentage (0-100): "))
    marks = float(input("Previous marks (0-100): "))

    if hours < 0 or not 0 <= attendance <= 100 or not 0 <= marks <= 100:
        print("Invalid input. Please enter values in the allowed ranges.")
    else:
        student = pd.DataFrame([{
            "study_hours": hours,
            "attendance": attendance,
            "previous_marks": marks
        }])

        result = model.predict(student)[0]
        print("\nStudent prediction:", result)

except ValueError:
    print("Please enter valid numeric values.")

print("\nPROJECT COMPLETED!")