import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import xgboost as xgb

from data_processing import load_data

def extract_features(df):
    X = df.drop(columns=['scalar_Classification'], errors='ignore')
    return X

CLASS_MAPPING = {
    0: ("Low Vegetation", "yellow"),
    1: ("Terrain", "gray"),
    2: ("Out Points", "blue"),
    3: ("Stem", "red"),
    4: ("Live Branches", "green"),
    5: ("Woody Branches", "orange"),
}

def save_model_plot(points, preds, title, out_path):
    print(f"Generating plot for {title}...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for cls_idx, (label_name, color) in CLASS_MAPPING.items():
        mask = (preds == cls_idx)
        if np.any(mask):
            # Using very small point size (s=0.1) since we are plotting many points
            ax.scatter(points[mask, 0], points[mask, 1], points[mask, 2], 
                       c=color, label=label_name, s=0.1, alpha=0.8)
    
    ax.set_title(title, y=-0.1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    # Adjust view angle to match screenshot
    ax.view_init(elev=30, azim=-45)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def save_legend(out_path):
    fig = plt.figure(figsize=(4, 6))
    ax = fig.add_subplot(111)
    ax.axis('off')
    legend_patches = [mpatches.Patch(color=color, label=label) for cls, (label, color) in CLASS_MAPPING.items()]
    ax.legend(handles=legend_patches, loc='center', frameon=False, title="Legend", title_fontsize='large')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

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
        print("Models not found! Please run `python src/benchmark.py` first to save all 3 models.")
        return
        
    print("Loading test data...")
    test_df = load_data(test_path)
    
    print(f"Using all {len(test_df)} points for visualization...")
        
    X_test = extract_features(test_df)
    
    x = test_df['x'].values
    y = test_df['y'].values
    z = test_df['z'].values
    points = np.vstack((x, y, z)).transpose()
    
    print("Running predictions...")
    preds_xgb = model_xgb.predict(xgb.DMatrix(X_test))
    
    preds_lgb_probs = model_lgb.predict(X_test)
    preds_lgb = np.argmax(preds_lgb_probs, axis=1)
    
    preds_cat = model_cat.predict(X_test).flatten()
    
    # Save individual high-res plots
    save_model_plot(points, preds_xgb, "XGBoost Segmentation", os.path.join(base_dir, 'plot_xgboost.png'))
    save_model_plot(points, preds_lgb, "LightGBM Segmentation", os.path.join(base_dir, 'plot_lightgbm.png'))
    save_model_plot(points, preds_cat, "CatBoost Segmentation", os.path.join(base_dir, 'plot_catboost.png'))
    
    # Save the legend separately
    save_legend(os.path.join(base_dir, 'plot_legend.png'))
    
    print("All high-resolution plots have been saved successfully to your project folder!")

if __name__ == '__main__':
    main()
