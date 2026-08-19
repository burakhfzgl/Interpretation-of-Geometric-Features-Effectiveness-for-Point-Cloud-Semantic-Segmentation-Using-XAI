import lightgbm as lgb

def train_lightgbm(X_train, y_train, params=None, num_classes=5):
    """
    Train a LightGBM model for point cloud semantic segmentation.
    Uses better hyperparameters to boost baseline performance.
    """
    if params is None:
        params = {
            'objective': 'multiclass',
            'num_class': num_classes,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 127,
            'max_depth': -1,
            'feature_fraction': 0.8,
            'min_data_in_leaf': 10,
            'n_jobs': -1,
            'verbose': -1
        }
        
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data, num_boost_round=150)
    return model
