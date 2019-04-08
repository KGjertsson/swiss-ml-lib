import csv
import time

from hyperopt import STATUS_OK, tpe, fmin, Trials, rand
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
import xgboost as xgb

from ..metrics import regression_metrics


# TODO: verify that hyperopt still works after all changes
# TODO: update hyperopt trainer to make use of model factories

def make_lasso(alpha=0.0005):
    return make_pipeline(RobustScaler(), Lasso(alpha=alpha, random_state=1))


def make_enet(alpha=0.0005, l1_ratio=.9):
    return make_pipeline(
        RobustScaler(),
        ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=3))


def make_kernel_ridge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5):
    return KernelRidge(alpha=alpha, kernel=kernel, degree=degree, coef0=coef0)


META_MODEL_TYPES = {
    'tpe': tpe.suggest,
    'random': rand.suggest
}

CHILD_MODEL_TYPES = {
    'xgboost': xgb.XGBRegressor,
    'lasso': make_lasso,
    'ENet': make_enet,
    'KRR': make_kernel_ridge,
    'GBoost': GradientBoostingRegressor
}

TRAIN_FUNS = {
    'mean_rmsle_cv': regression_metrics.mean_root_mean_squared_error_cv
}


class HyeroptTrainer:
    def __init__(self, train_data, train_labels, epochs, train_fun,
                 child_model_params, child_model_type='xgboost',
                 meta_model_type="tpe", n_folds=None, verbose=0):

        assert meta_model_type in META_MODEL_TYPES, \
            'Expected one of {} for argument meta_model_type'.format(
                META_MODEL_TYPES)
        assert child_model_type in CHILD_MODEL_TYPES, \
            'Expected one of {} for argument model_type'.format(
                CHILD_MODEL_TYPES)
        assert train_fun in TRAIN_FUNS, \
            'Expected one of {} for argument metric'.format(TRAIN_FUNS)

        self.train_data = train_data
        self.train_labels = train_labels
        self.epochs = epochs
        self.train_fun = TRAIN_FUNS[train_fun]
        self.domain_space = child_model_params
        self.child_model_class = CHILD_MODEL_TYPES[child_model_type]
        self.optimizer = META_MODEL_TYPES[meta_model_type]
        self.n_folds = n_folds
        self.verbose = verbose

        self.bayes_trials = Trials()
        self.iteration = 0
        self.best_parameter_setting = None
        self.best_parameter_setting_name = ''
        self.out_file = child_model_type + '_hyperopt_log.csv'

    def fit(self):
        self.prepare_out_results()
        self.best_parameter_setting = fmin(
            fn=self.objective, space=self.domain_space,
            algo=self.optimizer, max_evals=self.epochs,
            trials=self.bayes_trials, verbose=self.verbose)

    def objective(self, params):
        if 'max_depth' in params:
            params['max_depth'] = int(params['max_depth'])
        if 'n_estimators' in params:
            params['n_estimators'] = int(params['n_estimators'])
        if 'degree' in params:
            params['degree'] = int(params['degree'])

        self.iteration += 1
        print('iteration {}'.format(self.iteration))

        start_time = time.time()

        child_model = self.child_model_class(**params)
        train_args = [child_model, self.train_data, self.train_labels]
        if self.n_folds:
            train_args += [self.n_folds]

        hyperopt_loss = self.train_fun(*train_args)

        child_model = self.child_model_class(**params)
        child_model.fit(self.train_data.values, self.train_labels)
        train_pred = child_model.predict(self.train_data.values)
        training_loss = \
            np.sqrt(mean_squared_error(self.train_labels, train_pred))

        run_time = time.time() - start_time
        self.write_to_csv(
            hyperopt_loss, training_loss, self.iteration, run_time, params)

        return {'loss': hyperopt_loss,
                'params': params,
                'iteration': self.iteration,
                'train_time': run_time,
                'status': STATUS_OK}

    def prepare_out_results(self):
        with open(self.out_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['cross validation loss', 'training loss',
                             'iteration', 'train_time', 'params'])

    def write_to_csv(self, loss, training_loss, iteration, run_time, params):
        with open(self.out_file, 'a') as of_connection:
            writer = csv.writer(of_connection)
            writer.writerow([loss, training_loss, iteration, run_time, params])
