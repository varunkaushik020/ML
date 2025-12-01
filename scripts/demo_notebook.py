import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

dataset_paths = [
    'dataset/dataset1/diabetes.csv',
    'dataset/dataset3/heart_disease_uci.csv',
    'dataset/dataset4/chronic_disease_dataset.csv',
    'dataset/dataset6/dataset.csv'
]

datasets = {}

for path in dataset_paths:
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            dataset_name = os.path.basename(os.path.dirname(path))
            datasets[dataset_name] = df
            print(f"{dataset_name}: Shape {df.shape}")
        except Exception as e:
            print(f"Error loading {path}: {e}")
    else:
        print(f"File not found: {path}")

diabetes_df = datasets.get('dataset1')
diabetes_df.head()

print("Dataset shape:", diabetes_df.shape)
print("\nColumn names:")
print(diabetes_df.columns.tolist())

print("\nData types:")
print(diabetes_df.dtypes)

print("\nMissing values:")
print(diabetes_df.isnull().sum())

print("\nTarget distribution:")
print(diabetes_df['Outcome'].value_counts())

diabetes_df.describe()

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

diabetes_df['Outcome'].value_counts().plot(kind='bar', ax=axes[0, 0])
axes[0, 0].set_title('Distribution of Diabetes Outcome')
axes[0, 0].set_xlabel('Outcome (0=No, 1=Yes)')
axes[0, 0].set_ylabel('Count')

diabetes_df.boxplot(column='Age', by='Outcome', ax=axes[0, 1])
axes[0, 1].set_title('Age Distribution by Diabetes Outcome')

diabetes_df.boxplot(column='BMI', by='Outcome', ax=axes[1, 0])
axes[1, 0].set_title('BMI Distribution by Diabetes Outcome')

diabetes_df.boxplot(column='Glucose', by='Outcome', ax=axes[1, 1])
axes[1, 1].set_title('Glucose Distribution by Diabetes Outcome')

plt.tight_layout()
plt.show()

heart_df = datasets.get('dataset3')
heart_df.head()


print("Dataset shape:", heart_df.shape)
print("\nColumn names:")
print(heart_df.columns.tolist())

print("\nTarget distribution:")
print(heart_df['num'].value_counts())


fig, axes = plt.subplots(2, 2, figsize=(15, 10))

heart_df['num'].value_counts().plot(kind='bar', ax=axes[0, 0])
axes[0, 0].set_title('Distribution of Heart Disease')
axes[0, 0].set_xlabel('Heart Disease Severity (0=No, 1-4=Yes)')
axes[0, 0].set_ylabel('Count')

heart_df.boxplot(column='age', by='num', ax=axes[0, 1])
axes[0, 1].set_title('Age Distribution by Heart Disease Severity')

heart_df['chol'].hist(bins=30, ax=axes[1, 0])
axes[1, 0].set_title('Cholesterol Distribution')
axes[1, 0].set_xlabel('Cholesterol')
axes[1, 0].set_ylabel('Frequency')

heart_df['trestbps'].hist(bins=30, ax=axes[1, 1])
axes[1, 1].set_title('Resting Blood Pressure Distribution')
axes[1, 1].set_xlabel('Blood Pressure')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()


chronic_df = datasets.get('dataset4')
chronic_df.head()


print("Dataset shape:", chronic_df.shape)
print("\nColumn names:")
print(chronic_df.columns.tolist())

print("\nTarget distribution:")
print(chronic_df['target'].value_counts())

fig, axes = plt.subplots(2, 2, figsize=(15, 10))


chronic_df['target'].value_counts().plot(kind='bar', ax=axes[0, 0])
axes[0, 0].set_title('Distribution of Chronic Disease Target')
axes[0, 0].set_xlabel('Target')
axes[0, 0].set_ylabel('Count')

chronic_df.boxplot(column='age', by='target', ax=axes[0, 1])
axes[0, 1].set_title('Age Distribution by Disease Target')

chronic_df['bmi'].hist(bins=30, ax=axes[1, 0])
axes[1, 0].set_title('BMI Distribution')
axes[1, 0].set_xlabel('BMI')
axes[1, 0].set_ylabel('Frequency')

chronic_df['glucose_level'].hist(bins=30, ax=axes[1, 1])
axes[1, 1].set_title('Glucose Level Distribution')
axes[1, 1].set_xlabel('Glucose Level')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()


X = diabetes_df.drop(columns=['Outcome'])
y = diabetes_df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 10 Important Features:")
print(feature_importance.head(10))


plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
plt.title('Top 10 Feature Importances')
plt.show()