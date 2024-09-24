# data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Load the dataset from the CSV file.
    :param file_path: str, path to the dataset
    :return: pandas DataFrame
    """
    return pd.read_csv(file_path)

def preprocess_data(df):
    """
    Preprocess the dataset.
    - Drop unnecessary columns
    - Convert categorical labels into numeric
    :param df: pandas DataFrame
    :return: X (features), y (labels)
    """
    #Drop rows with NaN values in 'Species'
    df.dropna(subset=['Species'], inplace=True)

    # Separate features and labels
    X = df.drop(columns=["Species"])
    y = df["Species"].map({"setosa": 0, "versicolor": 1, "virginica": 2})

    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.
    :param X: features
    :param y: labels
    :param test_size: float, size of test set
    :param random_state: int, random seed for reproducibility
    :return: X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Example usage
    data_file_path = "./data/raw/Iris.csv"  # Adjusted path to your dataset
    df = load_data(data_file_path)
    
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Data preprocessed and split successfully.")
