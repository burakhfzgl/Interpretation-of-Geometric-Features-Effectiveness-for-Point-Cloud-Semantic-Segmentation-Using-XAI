import xgboost as xgb

def train_xgboost(X_train, y_train, params=None, num_classes=5):
    """
    Train an XGBoost model for point cloud semantic segmentation.
    Uses better hyperparameters to boost baseline performance.
    """
    if params is None:
        params = {
            'objective': 'multi:softmax',
            'num_class': num_classes,
            'eval_metric': 'mlogloss',
            'eta': 0.1,
            'max_depth': 8,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist' # faster training
        }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    # Increase number of boosting rounds for better accuracy
    model = xgb.train(params, dtrain, num_boost_round=150)
    return model
