# src/preprocess.py

import pandas as pd

def load_data(path="data/employees.csv"):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    # Handle missing values (if any)
    df = df.dropna()

    # Encode categorical variables
    df = pd.get_dummies(df, columns=['Department'], drop_first=True)

    return df

def split_features_target(df):
    X = df.drop('Performance', axis=1)
    y = df['Performance']
    return X, y