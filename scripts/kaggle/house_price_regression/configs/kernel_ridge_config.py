from sml.regression.data_management.top_4_percent_cleaner import \
    Top4PercentDataCleaner
from sml.regression.model_factories.scikit_regression_factory import \
    ScikitRegressionFactory
from sml.regression.train.regression_trainer import RegressionTrainer

MODEL_TRAINER_CLS = RegressionTrainer
MODEL_FACTORY_CLS = ScikitRegressionFactory
MODEL_KWARGS = {'model_type': 'KernelRidge',
                'alpha': 0.6,
                'kernel': 'polynomial',
                'degree': 2,
                'coef0': 2.5}
DATA_CLEANER_CLS = Top4PercentDataCleaner
