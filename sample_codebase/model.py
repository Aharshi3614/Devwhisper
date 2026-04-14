import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess(df):
    df = df.dropna()
    df['date'] = pd.to_datetime(df['date'])
    X = df.drop(columns=['target'])
    y = df['target']
    return X, y

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model, X_test, y_test

def train_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    acc = accuracy_score(y_test, predictions)
    print(f"RMSE: {rmse:.4f}")
    print(f"Accuracy: {acc:.4f}")
    return rmse, acc

def save_model(model, path="model.pkl"):
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def load_model(path="model.pkl"):
    model = joblib.load(path)
    return model