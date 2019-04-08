from sml.regression.data_management.top_4_percent_cleaner import \
    Top4PercentDataCleaner
from sml.regression.model_factories.scikit_regression_factory import \
    ScikitRegressionFactory
from sml.regression.train.regression_trainer import RegressionTrainer

MODEL_TRAINER_CLS = RegressionTrainer
MODEL_FACTORY_CLS = ScikitRegressionFactory
MODEL_KWARGS = {'model_type': 'LightGBM',
                'objective': 'regression',
                'num_leaves': 5,
                'learning_rate': 0.05,
                'n_estimators': 720,
                'max_bin': 55,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'feature_fraction': 0.2319,
                'feature_fraction_seed': 9,
                'bagging_seed': 9,
                'min_data_in_leaf': 6,
                'min_sum_hessian_in_leaf': 11}
DATA_CLEANER_CLS = Top4PercentDataCleaner
