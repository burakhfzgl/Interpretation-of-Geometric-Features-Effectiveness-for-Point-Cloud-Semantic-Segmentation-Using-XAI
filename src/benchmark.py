import time
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np

from data_processing import load_data
from models.train_xgboost import train_xgboost
from models.train_catboost import train_catboost
from models.train_lightgbm import train_lightgbm

def extract_features_and_labels(df):
    """
    Separates the DataFrame into features (X) and labels (y).
    """
    if 'scalar_Classification' not in df.columns:
        raise ValueError("Target column 'scalar_Classification' not found in dataset.")
        
    y = df['scalar_Classification']
    X = df.drop(columns=['scalar_Classification'])
    return X, y

def evaluate_and_store(model_name, y_true, y_pred, train_time, results_list):
    acc = accuracy_score(y_true, y_pred)
    # Using 'weighted' average to account for class imbalance, typical in point cloud metrics
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    results_list.append({
        "Classifier": model_name,
        "Dataset": "RMIT",
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
        "OA (Accuracy)": round(acc, 4),
        "Train Time (s)": round(train_time, 2)
    })
    
    print(f"[{model_name}] OA: {acc:.4f} | F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")

def main():
    print("Loading datasets...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, 'data', 'processed', 'train.ply')
    test_path = os.path.join(base_dir, 'data', 'processed', 'test.ply')
    
    train_df = load_data(train_path)
    test_df = load_data(test_path)
    
    print("Extracting features...")
    X_train, y_train_raw = extract_features_and_labels(train_df)
    X_test, y_test_raw = extract_features_and_labels(test_df)
    
    print("Encoding labels to 0-indexed values...")
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)
    
    num_classes = len(le.classes_)
    print(f"Detected {num_classes} classes: {le.classes_}")
    
    results = []

    # 1. XGBoost
    print("\nTraining XGBoost...")
    start_time = time.time()
    try:
        model_xgb = train_xgboost(X_train, y_train, num_classes=num_classes)
        train_time_xgb = time.time() - start_time
        
        import xgboost as xgb
        import joblib
        
        # Save the model for visualization
        models_dir = os.path.join(base_dir, 'models', 'saved')
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(model_xgb, os.path.join(models_dir, 'xgboost_model.joblib'))
        
        dtest = xgb.DMatrix(X_test)
        preds_xgb = model_xgb.predict(dtest)
        
        evaluate_and_store("XGBoost", y_test, preds_xgb, train_time_xgb, results)
    except Exception as e:
        print(f"XGBoost failed: {e}")

    # 2. CatBoost
    print("\nTraining CatBoost...")
    start_time = time.time()
    try:
        model_cat = train_catboost(X_train, y_train)
        train_time_cat = time.time() - start_time
        import joblib
        joblib.dump(model_cat, os.path.join(models_dir, 'catboost_model.joblib'))
        
        preds_cat = model_cat.predict(X_test)
        
        evaluate_and_store("CatBoost", y_test, preds_cat, train_time_cat, results)
    except Exception as e:
        print(f"CatBoost failed: {e}")

    # 3. LightGBM
    print("\nTraining LightGBM...")
    start_time = time.time()
    try:
        model_lgb = train_lightgbm(X_train, y_train, num_classes=num_classes)
        train_time_lgb = time.time() - start_time
        import joblib
        joblib.dump(model_lgb, os.path.join(models_dir, 'lightgbm_model.joblib'))
        
        preds_lgb_probs = model_lgb.predict(X_test)
        
        preds_lgb = np.argmax(preds_lgb_probs, axis=1)
        
        evaluate_and_store("LightGBM", y_test, preds_lgb, train_time_lgb, results)
    except Exception as e:
        print(f"LightGBM failed: {e}")

    # Display Results
    results_df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("BENCHMARK RESULTS".center(80))
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)

if __name__ == '__main__':
    main()
