# from sml.regression.data_management.top_4_percent_cleaner import \
#     Top4PercentDataCleaner
from sml.regression.data_management.ming_data_cleaner import MingDataCleaner
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
        'Lasso': {'alpha': 0.0005,
                  'random_state': 1},
        'ElasticNet': {'alpha': 0.0005,
                       'l1_ratio': .9,
                       'random_state': 3},
        'KernelRidge': {'alpha': 0.6,
                        'kernel': 'polynomial',
                        'degree': 2,
                        'coef0': 2.5},
        'GradientBoostingRegressor': {'n_estimators': 3000,
                                      'learning_rate': 0.05,
                                      'max_depth': 4,
                                      'max_features': 'sqrt',
                                      'min_samples_leaf': 15,
                                      'min_samples_split': 10}
    }
}
DATA_CLEANER_CLS = MingDataCleaner
