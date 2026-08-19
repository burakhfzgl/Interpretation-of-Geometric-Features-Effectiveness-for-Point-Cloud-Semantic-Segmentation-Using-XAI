import os
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import xgboost as xgb

from data_processing import load_data

def extract_features_and_labels(df):
    y = df['scalar_Classification']
    X = df.drop(columns=['scalar_Classification'], errors='ignore')
    return X, y

CLASS_MAPPING = {
    0: "Low_Vegetation",
    1: "Terrain",
    2: "Out_Points",
    3: "Stem",
    4: "Live_Branches",
    5: "Woody_Branches",
}

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_path = os.path.join(base_dir, 'data', 'processed', 'test.ply')
    models_dir = os.path.join(base_dir, 'models', 'saved')
    
    print("Loading models...")
    try:
        model_xgb = joblib.load(os.path.join(models_dir, 'xgboost_model.joblib'))
        model_cat = joblib.load(os.path.join(models_dir, 'catboost_model.joblib'))
        model_lgb = joblib.load(os.path.join(models_dir, 'lightgbm_model.joblib'))
    except FileNotFoundError:
        print("Models not found! Please run `python src/benchmark.py` first.")
        return
        
    print("Loading test data...")
    test_df = load_data(test_path)
    X_test, y_test = extract_features_and_labels(test_df)
    
    sample_size = 5000
    print(f"Subsampling {sample_size} points for SHAP analysis...")
    indices = np.random.choice(len(X_test), size=sample_size, replace=False)
    X_sample = X_test.iloc[indices]
    y_sample = y_test.iloc[indices].values
    
    # Find one representative point for each class for Local Analysis
    num_classes = 6
    representative_points = {}
    for c in range(num_classes):
        # Find index in the sample where the true class is `c`
        idxs = np.where(y_sample == c)[0]
        if len(idxs) > 0:
            representative_points[c] = idxs[0]
            
    models_to_evaluate = [
        ("XGBoost", model_xgb),
        ("CatBoost", model_cat),
        ("LightGBM", model_lgb)
    ]
    
    for model_name, model in models_to_evaluate:
        print(f"\n{'='*40}")
        print(f"Analyzing {model_name}...")
        print(f"{'='*40}")
        
        try:
            if model_name == "CatBoost":
                import catboost
                pool = catboost.Pool(X_sample)
                # CatBoost's native C++ SHAP explainer (avoids Python segfaults)
                shap_raw = model.get_feature_importance(pool, type='ShapValues')
                
                # CatBoost returns (N, F+1, C) or (N, C, F+1)
                if shap_raw.shape[1] == num_classes:
                    # Shape is (N, C, F+1)
                    shap_list = [shap_raw[:, c, :-1] for c in range(num_classes)]
                    base_values = shap_raw[0, :, -1]
                else:
                    # Shape is (N, F+1, C)
                    shap_list = [shap_raw[:, :-1, c] for c in range(num_classes)]
                    base_values = shap_raw[0, -1, :]
            else:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(X_sample)
                is_list = isinstance(shap_values.values, list)
                if is_list:
                    shap_list = shap_values.values
                    base_values = explainer.expected_value
                else:
                    shap_list = [shap_values[:, :, c].values for c in range(num_classes)]
                    base_values = explainer.expected_value
            
            # 1. GLOBAL ANALYSIS
            print(f"[{model_name}] Generating Global Summary Plot...")
            out_path = os.path.join(base_dir, f'shap_global_summary_{model_name.lower()}.png')
            plt.figure(figsize=(12, 10))
            
            # Draw summary plot
            shap.summary_plot(shap_list, X_sample, plot_type="bar", show=False, class_names=list(CLASS_MAPPING.values()), max_display=len(X_sample.columns))
            
            # Override the default x-axis label
            plt.xlabel("Mean Absolute SHAP Value (Feature Importance)", fontsize=12)
            
            # Force legend to lower right corner
            plt.legend(loc="lower right", framealpha=0.9, title="Classes")
            
            plt.title(f"Global Feature Importance - {model_name}")
            plt.tight_layout()
            plt.savefig(out_path, dpi=300)
            plt.close()
            print(f"Saved {out_path}")
            
            # 2. LOCAL ANALYSIS
            print(f"[{model_name}] Generating Local Decision Plots...")
            for class_idx, point_idx in representative_points.items():
                class_name = CLASS_MAPPING.get(class_idx, f"Class_{class_idx}")
                out_path = os.path.join(base_dir, f'shap_local_{model_name.lower()}_{class_name}.png')
                plt.figure(figsize=(12, 10))
                
                base_value = base_values[class_idx]
                shap_vals_for_point = shap_list[class_idx][point_idx]
                
                shap.decision_plot(base_value, shap_vals_for_point, X_sample.iloc[point_idx], show=False)
                plt.title(f"Local Decision Plot - {model_name} (Predicting {class_name.replace('_', ' ')})")
                plt.tight_layout()
                plt.savefig(out_path, dpi=300)
                plt.close()
                print(f"Saved {out_path}")
                
        except Exception as e:
            print(f"Failed to generate SHAP for {model_name}: {e}")
            
    print("\nSHAP analysis complete! Plots are saved in your project root.")

if __name__ == '__main__':
    main()
