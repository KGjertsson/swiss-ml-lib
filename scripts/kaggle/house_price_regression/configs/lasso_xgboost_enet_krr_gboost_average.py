from sml.regression.data_management.top_4_percent_cleaner import \
    Top4PercentDataCleaner
from sml.regression.models.average_regression import MergeByAverageRegression
from sml.regression.model_factories.ensemble_factory import \
    EnsembleRegressionFactory
from sml.regression.model_factories.scikit_regression_factory import \
    ScikitRegressionFactory
from sml.regression.train.regression_trainer import RegressionTrainer

MODEL_TRAINER_CLS = RegressionTrainer
MODEL_FACTORY_CLS = EnsembleRegressionFactory
MODEL_KWARGS = {
    'ensemble_cls': MergeByAverageRegression,
    'child_factory_cls': ScikitRegressionFactory,
    'model_data': {
        'Lasso': {'alpha': 0.0004620609545131168},
        'ElasticNet': {'alpha': 0.0006514763771079174},
        'KernelRidge': {'alpha': 0.9813351184821618,
                        'coef0': 6.373971377335225,
                        'degree': 2,
                        'kernel': 'polynomial'},
        'GradientBoostingRegressor': {'learning_rate': 0.05952166814322882,
                                      'loss': 'huber',
                                      'max_depth': 2,
                                      'max_features': 'sqrt',
                                      'min_samples_leaf': 15,
                                      'min_samples_split': 10,
                                      'n_estimators': 1446,
                                      'random_state': 5},
        'XGBoost': {'colsample_bytree': 0.4603,
                    'gamma': 0.0004742746687883887,
                    'learning_rate': 0.006806325375209027,
                    'max_depth': 3, 'min_child_weight': 1.7817,
                    'n_estimators': 4795,
                    'n_jobs': -1,
                    'random_state': 7,
                    'reg_alpha': 0.464,
                    'reg_lambda': 0.8571,
                    'silent': 1,
                    'subsample': 0.5213}
    }
}
DATA_CLEANER_CLS = Top4PercentDataCleaner
