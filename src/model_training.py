import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    # Drop rows with NaN values in 'Species' column
    df.dropna(subset=['Species'], inplace=True)
    
    X = df.drop(columns=["Species"])
    y = df["Species"].map({"setosa": 0, "versicolor": 1, "virginica": 2})

    return X, y

def train_model(X, y):
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

if __name__ == "__main__":
    data_file_path = "./data/raw/Iris.csv"
    df = load_data(data_file_path)

    # Check unique values in 'species' column
    print("Unique values in 'Species' column before mapping:\n", df['Species'].unique())

    X, y = preprocess_data(df)

    # Check for NaN values after preprocessing
    print("NaN values in y after preprocessing:\n", y.isnull().sum())

    if y.isnull().any():
        raise ValueError("Cannot train the model because y contains NaN values.")

    model = train_model(X, y)
    print("Model trained successfully.")