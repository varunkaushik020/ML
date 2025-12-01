import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib


def load_and_preprocess_data():
    diabetes_df = pd.read_csv('dataset/dataset1/diabetes.csv')
    return diabetes_df


def prepare_features(df):
    if 'Outcome' in df.columns:
        X = df.drop(['Outcome'], axis=1)
        y = df['Outcome']
    else:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

    numeric_columns = X.select_dtypes(include=[np.number]).columns
    X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].mean())

    return X, y


def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)

    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_accuracy = accuracy_score(y_test, lr_pred)

    if rf_accuracy >= lr_accuracy:
        best_model = rf_model
        model_name = "RandomForest"
        best_accuracy = rf_accuracy
    else:
        best_model = lr_model
        model_name = "LogisticRegression"
        best_accuracy = lr_accuracy

    return best_model, scaler, model_name, best_accuracy


def save_model(model, scaler, model_name):
    if not os.path.exists('models'):
        os.makedirs('models')

    model_path = f'models/chronic_disease_model_{model_name.lower()}.pkl'
    joblib.dump(model, model_path)

    scaler_path = 'models/scaler.pkl'
    joblib.dump(scaler, scaler_path)


def main():
    df = load_and_preprocess_data()
    X, y = prepare_features(df)
    best_model, scaler, model_name, accuracy = train_models(X, y)
    save_model(best_model, scaler, model_name)


if __name__ == "__main__":
    main()
