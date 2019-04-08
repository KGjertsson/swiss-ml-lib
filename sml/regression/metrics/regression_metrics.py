import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_root_mean_squared_error_cv(_model, _train, _y_train, n_folds=5):
    kf = KFold(
        n_folds, shuffle=True, random_state=42).get_n_splits(_train.values)
    rmse = np.sqrt(
        -cross_val_score(_model, _train.values, _y_train,
                         scoring="neg_mean_squared_error", cv=kf, verbose=1))
    return np.mean(rmse)
