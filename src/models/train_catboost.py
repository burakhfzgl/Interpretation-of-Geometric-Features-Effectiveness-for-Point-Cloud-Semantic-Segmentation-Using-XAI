from catboost import CatBoostClassifier

def train_catboost(X_train, y_train, params=None):
    """
    Train a CatBoost model for point cloud semantic segmentation.
    Uses better hyperparameters to boost baseline performance.
    """
    if params is None:
        params = {
            'iterations': 200,
            'learning_rate': 0.1,
            'depth': 8,
            'loss_function': 'MultiClass',
            'task_type': 'CPU'
        }
        
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, verbose=50)
    return model
