import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
print("📊 Loading Iris Dataset...")
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Create DataFrame for better visualization
df = pd.DataFrame(X, columns=feature_names)
df['species'] = y
df['species_name'] = [target_names[i] for i in y]

print(f"Dataset shape: {X.shape}")
print(f"Features: {feature_names}")
print(f"Target classes: {target_names}")

# Check for missing values
print(f"\n🔍 Missing values: {df.isnull().sum().sum()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📈 Training set: {X_train.shape[0]} samples")
print(f"📊 Test set: {X_test.shape[0]} samples")

# Train Decision Tree
print("\n🌲 Training Decision Tree Classifier...")
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Accuracy: {accuracy:.2%}")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix - Iris Classification')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_iris.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature Importance
plt.figure(figsize=(8, 4))
importance = model.feature_importances_
feature_imp = pd.DataFrame({'feature': feature_names, 'importance': importance})
feature_imp = feature_imp.sort_values('importance', ascending=True)

plt.barh(feature_imp['feature'], feature_imp['importance'])
plt.title('Feature Importance - Decision Tree')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance_iris.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Task 1 Complete! Classical ML model trained successfully.")