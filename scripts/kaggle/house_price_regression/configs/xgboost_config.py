from sml.regression.data_management.top_4_percent_cleaner import \
    Top4PercentDataCleaner
from sml.regression.data_management.ming_data_cleaner import MingDataCleaner
from sml.regression.model_factories.scikit_regression_factory import \
    ScikitRegressionFactory
from sml.regression.train.regression_trainer import RegressionTrainer

MODEL_TRAINER_CLS = RegressionTrainer
MODEL_FACTORY_CLS = ScikitRegressionFactory

MODEL_KWARGS = {'model_type': 'XGBoost',
                'colsample_bytree': 0.4603,
                'gamma': 0.0004742746687883887,
                'learning_rate': 0.006806325375209027,
                'max_depth': 3,
                'min_child_weight': 1.7817,
                'n_estimators': 4795,
                'n_jobs': -1,
                'random_state': 7,
                'reg_alpha': 0.464,
                'reg_lambda': 0.8571,
                'silent': 1,
                'subsample': 0.5213}

DATA_CLEANER_CLS = MingDataCleaner
