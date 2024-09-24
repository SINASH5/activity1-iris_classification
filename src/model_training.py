import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_data(file_path):
    """
    Load the dataset from the CSV file.
    :param file_path: str, path to the dataset
    :return: pandas DataFrame
    """
    return pd.read_csv(file_path)

def train_model(X, y):
    """
    Train a Random Forest model.
    :param X: features
    :param y: labels
    :return: trained model
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

def main():
    data_file_path = "./data/raw/Iris.csv"  # Adjust this path if needed
    df = load_data(data_file_path)

    # Check for NaN values in features and labels
    if df.isnull().sum().sum() > 0:
        print("Data contains NaN values. Dropping NaNs...")
        df.dropna(inplace=True)  # Remove rows with NaN values

    # Separate features and labels
    X = df.drop(columns=["Species"])  # Changed to "Species"
    y = df["Species"]  # Changed to "Species"

    # Check for NaN values in y
    if y.isnull().sum() > 0:
        print("Target variable contains NaN values. Removing NaN entries...")
        df.dropna(subset=["Species"], inplace=True)  # Changed to "Species"
        y = df["Species"]  # Update y after dropping NaNs

    # Train the model
    model = train_model(X, y)

    # Predictions and accuracy check
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    print(f"Model trained with accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()