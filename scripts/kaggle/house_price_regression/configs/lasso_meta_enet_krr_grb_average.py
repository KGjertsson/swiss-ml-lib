# from sml.regression.data_management.top_4_percent_cleaner import \
#     Top4PercentDataCleaner
from sml.regression.data_management.ming_data_cleaner import MingDataCleaner
from sml.regression.models.averate_meta_regression import \
    StackingAveragedModels
from sml.regression.model_factories.ensemble_meta_factory import \
    MetaEnsembleFactory
from sml.regression.model_factories.scikit_regression_factory import \
    ScikitRegressionFactory
from sml.regression.train.regression_trainer import RegressionTrainer

MODEL_TRAINER_CLS = RegressionTrainer
MODEL_FACTORY_CLS = MetaEnsembleFactory
MODEL_KWARGS = {
    'ensemble_cls': StackingAveragedModels,
    'child_factory_cls': ScikitRegressionFactory,
    'meta_model_data': {
        'Lasso': {'alpha': 0.0004620609545131168,
                  'random_state': 1}
    },
    'model_data': {
        'Lasso': {'alpha': 0.0004620609545131168,
                  'random_state': 1},
        'ElasticNet': {'alpha': 0.0006514763771079174,
                       'l1_ratio': .9,
                       'random_state': 3},
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
                                      'random_state': 5}
    }
}
# DATA_CLEANER_CLS = Top4PercentDataCleaner
DATA_CLEANER_CLS = MingDataCleaner
