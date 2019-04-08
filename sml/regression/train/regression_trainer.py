from time import time

import numpy as np
import pandas as pd

from ..metrics import regression_metrics


class RegressionTrainer:

    def __init__(self,
                 model_factory_cls,
                 model_kwargs,
                 data_root_dir,
                 data_cleaner_cls,
                 verbose=1):
        self.model_factory_cls = model_factory_cls
        self.model_kwargs = model_kwargs
        self.model = None

        self.data_root_dir = data_root_dir
        self.data_cleaner_cls = data_cleaner_cls
        self.verbose = verbose

        self.train_data_cleaned = None
        self.train_targets_cleaned = None
        self.test_data_cleaned = None
        self.test_ids = None

    def __call__(self, actions):
        self.configure_data()
        self.configure_model()

        for action in actions:
            getattr(self, action)()

    def configure_data(self):
        t_0 = time()
        if self.verbose:
            print('Cleaning data')
        cleaner = self.data_cleaner_cls(self.data_root_dir, self.verbose)
        cleaner()
        if self.verbose:
            print('Done ({} s)'.format(time() - t_0))
        self.train_data_cleaned = cleaner.train
        self.train_targets_cleaned = cleaner.y_train
        self.test_data_cleaned = cleaner.test
        self.test_ids = cleaner.test_ids

    def configure_model(self):
        self.model = self.model_factory_cls(**self.model_kwargs)()

    def cross_eval(self):
        t_0 = time()
        if self.verbose:
            print('Performing cross-validation')

        # TODO: make selection of objective function dynamic
        root_mean_squared_error = \
            regression_metrics.mean_root_mean_squared_error_cv(
                self.model,
                self.train_data_cleaned,
                self.train_targets_cleaned)

        if self.verbose:
            print('Done ({} s)'.format(time() - t_0))
            print('Cross validation root mean squared error: {}'.format(
                root_mean_squared_error))

    def fit(self):
        self.model.fit(
            self.train_data_cleaned.values, self.train_targets_cleaned)

    def predict(self):
        # TODO
        pass

    def make_submission(self):
        t_0 = time()
        if self.verbose:
            print('Making submission')
        predictions = self.model.predict(self.test_data_cleaned.values)
        predictions = np.expm1(predictions)

        sub = pd.DataFrame()
        sub['Id'] = self.test_ids
        sub['SalePrice'] = predictions
        sub.to_csv('submission.csv', index=False)
        if self.verbose:
            print('Done ({} s)'.format(time() - t_0))
