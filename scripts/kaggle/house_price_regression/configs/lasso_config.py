# from sml.regression.data_management.top_4_percent_cleaner import \
#     Top4PercentDataCleaner
from sml.regression.data_management.ming_data_cleaner import MingDataCleaner

from sml.regression.model_factories.scikit_regression_factory import \
    ScikitRegressionFactory
from sml.regression.train.regression_trainer import RegressionTrainer

MODEL_TRAINER_CLS = RegressionTrainer
MODEL_FACTORY_CLS = ScikitRegressionFactory
MODEL_KWARGS = {'model_type': 'Lasso', 'alpha': 0.0004620609545131168}
# DATA_CLEANER_CLS = Top4PercentDataCleaner
DATA_CLEANER_CLS = MingDataCleaner
