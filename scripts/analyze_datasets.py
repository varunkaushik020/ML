import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os


plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_and_explore_datasets():
    print("=== Loading and Exploring Datasets ===")

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
                print(f"\n{dataset_name}:")
                print(f"  Shape: {df.shape}")
                print(f"  Columns: {list(df.columns)}")
                print(f"  Missing values: {df.isnull().sum().sum()}")
            except Exception as e:
                print(f"Error loading {path}: {e}")
        else:
            print(f"File not found: {path}")

    return datasets


def analyze_diabetes_data(df):

    print("\n=== Diabetes Dataset Analysis ===")

    print(f"Dataset shape: {df.shape}")
    print(f"Column names: {list(df.columns)}")

    print(f"Missing values: {df.isnull().sum().sum()}")

    print(f"Target distribution:\n{df['Outcome'].value_counts()}")

    print("\nBasic statistics:")
    print(df.describe())

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    df['Outcome'].value_counts().plot(kind='bar', ax=axes[0, 0])
    axes[0, 0].set_title('Distribution of Diabetes Outcome')
    axes[0, 0].set_xlabel('Outcome (0=No, 1=Yes)')
    axes[0, 0].set_ylabel('Count')

    df.boxplot(column='Age', by='Outcome', ax=axes[0, 1])
    axes[0, 1].set_title('Age Distribution by Diabetes Outcome')

    if 'BMI' in df.columns:
        df.boxplot(column='BMI', by='Outcome', ax=axes[1, 0])
        axes[1, 0].set_title('BMI Distribution by Diabetes Outcome')

    df.boxplot(column='Glucose', by='Outcome', ax=axes[1, 1])
    axes[1, 1].set_title('Glucose Distribution by Diabetes Outcome')

    plt.tight_layout()
    plt.savefig('diabetes_analysis.png')
    plt.show()

    return df


def analyze_heart_disease_data(df):

    print("\n=== Heart Disease Dataset Analysis ===")

    print(f"Dataset shape: {df.shape}")
    print(f"Column names: {list(df.columns)}")

    print(f"Missing values: {df.isnull().sum().sum()}")

    if 'num' in df.columns:
        print(f"Target distribution:\n{df['num'].value_counts()}")

    numeric_columns = df.select_dtypes(include=[np.number]).columns
    print(f"\nNumeric columns: {list(numeric_columns)}")
    print("\nBasic statistics:")
    print(df[numeric_columns].describe())

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    if 'num' in df.columns:
        df['num'].value_counts().plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Distribution of Heart Disease')
        axes[0, 0].set_xlabel('Heart Disease Severity (0=No, 1-4=Yes)')
        axes[0, 0].set_ylabel('Count')


    if 'age' in df.columns and 'num' in df.columns:
        df.boxplot(column='age', by='num', ax=axes[0, 1])
        axes[0, 1].set_title('Age Distribution by Heart Disease Severity')


    if 'chol' in df.columns:
        df['chol'].hist(bins=30, ax=axes[1, 0])
        axes[1, 0].set_title('Cholesterol Distribution')
        axes[1, 0].set_xlabel('Cholesterol')
        axes[1, 0].set_ylabel('Frequency')

    if 'trestbps' in df.columns:
        df['trestbps'].hist(bins=30, ax=axes[1, 1])
        axes[1, 1].set_title('Resting Blood Pressure Distribution')
        axes[1, 1].set_xlabel('Blood Pressure')
        axes[1, 1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('heart_disease_analysis.png')
    plt.show()

    return df


def analyze_chronic_disease_data(df):
    print("\n=== Chronic Disease Dataset Analysis ===")

    print(f"Dataset shape: {df.shape}")
    print(f"Column names: {list(df.columns)}")

    print(f"Missing values: {df.isnull().sum().sum()}")

    if 'target' in df.columns:
        print(f"Target distribution:\n{df['target'].value_counts()}")

    print("\nBasic statistics:")
    print(df.describe())

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    if 'target' in df.columns:
        df['target'].value_counts().plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Distribution of Chronic Disease Target')
        axes[0, 0].set_xlabel('Target')
        axes[0, 0].set_ylabel('Count')

    if 'age' in df.columns and 'target' in df.columns:
        df.boxplot(column='age', by='target', ax=axes[0, 1])
        axes[0, 1].set_title('Age Distribution by Disease Target')

    if 'bmi' in df.columns:
        df['bmi'].hist(bins=30, ax=axes[1, 0])
        axes[1, 0].set_title('BMI Distribution')
        axes[1, 0].set_xlabel('BMI')
        axes[1, 0].set_ylabel('Frequency')

    if 'glucose_level' in df.columns:
        df['glucose_level'].hist(bins=30, ax=axes[1, 1])
        axes[1, 1].set_title('Glucose Level Distribution')
        axes[1, 1].set_xlabel('Glucose Level')
        axes[1, 1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('chronic_disease_analysis.png')
    plt.show()

    return df


def prepare_data_for_modeling(df, target_column):
    
    print(f"\n=== Preparing Data for Modeling ===")

    if target_column not in df.columns:
        print(f"Target column '{target_column}' not found in dataset")
        return None, None

    X = df.drop(columns=[target_column])
    y = df[target_column]

    categorical_columns = X.select_dtypes(include=['object']).columns
    numerical_columns = X.select_dtypes(include=[np.number]).columns

    print(f"Categorical columns: {list(categorical_columns)}")
    print(f"Numerical columns: {list(numerical_columns)}")

    X_encoded = X.copy()
    label_encoders = {}

    for col in categorical_columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    for col in numerical_columns:
        if X_encoded[col].isnull().sum() > 0:
            X_encoded[col].fillna(X_encoded[col].median(), inplace=True)

    print(f"Final dataset shape: {X_encoded.shape}")
    print(f"Target distribution:\n{y.value_counts()}")

    return X_encoded, y


def train_simple_model(X, y):
   
    print(f"\n=== Training Simple Model ===")

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
    plt.savefig('confusion_matrix.png')
    plt.show()

    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Important Features:")
    print(feature_importance.head(10))

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
    plt.title('Top 10 Feature Importances')
    plt.savefig('feature_importance.png')
    plt.show()

    return model, scaler


def main():

    print("Chronic Disease Dataset Analysis")
    print("=" * 40)

    datasets = load_and_explore_datasets()


    for name, df in datasets.items():
        print(f"\nAnalyzing {name}...")

        if name == "dataset1":
            df_processed = analyze_diabetes_data(df)
            X, y = prepare_data_for_modeling(df_processed, 'Outcome')
        elif name == "dataset3":
            df_processed = analyze_heart_disease_data(df)
            X, y = prepare_data_for_modeling(df_processed, 'num')
        elif name == "dataset4":
            df_processed = analyze_chronic_disease_data(df)
            X, y = prepare_data_for_modeling(df_processed, 'target')
        elif name == "dataset6":
      
            print("Dataset6 has a complex structure with many columns")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            continue
        else:
            continue


        if X is not None and y is not None and len(y) > 10:
            if len(np.unique(y)) > 1:
                model, scaler = train_simple_model(X, y)
                print(f"Model trained successfully for {name}")
            else:
                print(f"Not enough classes for classification in {name}")
        else:
            print(f"Insufficient data for modeling in {name}")


if __name__ == "__main__":
    main()
