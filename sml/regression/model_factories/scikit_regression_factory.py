import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
import xgboost as xgb

MODEL_MAKERS = ('Lasso', 'ElasticNet')

MODEL_TYPES = {
    'Lasso': Lasso,
    'ElasticNet': ElasticNet,
    'KernelRidge': KernelRidge,
    'GradientBoostingRegressor': GradientBoostingRegressor,
    'XGBoost': xgb.XGBRegressor,
    'LightGBM': lgb.LGBMRegressor
}


class ScikitRegressionFactory:

    def __init__(self, model_type, **kwargs):
        super().__init__()
        self.model_class = MODEL_TYPES[model_type]
        self.model_maker = model_type in MODEL_MAKERS
        self.kwargs = kwargs
        self.model = None

    def __call__(self):
        if self.model_maker:
            self.model = \
                make_pipeline(RobustScaler(), self.model_class(**self.kwargs))
        else:
            self.model = self.model_class(**self.kwargs)
        return self.model
