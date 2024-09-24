import pandas as pd
import joblib

def load_model(model_path):
    return joblib.load(model_path)

def make_prediction(model, input_data):
    # Ensure input_data is a DataFrame and has the same columns as the training data
    input_data = input_data.drop(columns=["Id"], errors='ignore')  # Drop 'Id' column if it exists
    return model.predict(input_data)

def main():
    model_path = "./models/iris_model.pkl"  # Adjust path if necessary
    model = load_model(model_path)

    # Example input data (should be in the same format as training data)
    # This should ideally come from user input or another source
    input_data = pd.DataFrame({
        "SepalLengthCm": [5.1],
        "SepalWidthCm": [3.5],
        "PetalLengthCm": [1.4],
        "PetalWidthCm": [0.2]
    })

    prediction = make_prediction(model, input_data)
    print(f"Predicted class: {prediction[0]}")

if __name__ == "__main__":
    main()

