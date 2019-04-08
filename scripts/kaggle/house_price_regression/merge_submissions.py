import pandas as pd

stacked = pd.read_csv('submission_stacked.csv')
xgb = pd.read_csv('submission_xgboost.csv')
lgbm = pd.read_csv('submission_lgbm.csv')

ensemble = stacked['SalePrice'] * 0.7 + xgb['SalePrice'] * 0.15 + lgbm[
    'SalePrice'] * 0.15

stacked['SalePrice'] = ensemble.values

stacked.to_csv('ensemble_submission.csv', index=False)
