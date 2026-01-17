import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load the dataset (Built-in to scikit-learn, so no external CSV needed)
print("Loading data...")
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train the model
print("Training Decision Tree...")
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 4. Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 5. Save the model
joblib.dump(model, 'iris_model.pkl')
print("Model saved as 'iris_model.pkl'")
