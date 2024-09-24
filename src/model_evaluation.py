# src/model_evaluation.py

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

def evaluate_model(y_true, y_pred):
    """
    Evaluate the performance of the model.
    
    Parameters:
    y_true : array-like, true labels
    y_pred : array-like, predicted labels
    
    Returns:
    dict : a dictionary with evaluation metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def load_model(model_path):
    """
    Load a saved model from a specified path.
    
    Parameters:
    model_path : str, path to the model file
    
    Returns:
    model : the loaded model
    """
    model = joblib.load(model_path)
    return model
