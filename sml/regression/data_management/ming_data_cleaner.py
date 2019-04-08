from pathlib import Path

import numpy as np
import pandas as pd

from .abstract_data_cleaner import AbstractDataCleaner


class MingDataCleaner(AbstractDataCleaner):

    def __init__(self, data_root_dir, verbose):
        super().__init__(data_root_dir, verbose)

        self.train = pd.read_csv(Path(self.data_root_dir, 'train.csv'))
        self.test = pd.read_csv(Path(self.data_root_dir, 'test.csv'))

    def __call__(self, *args, **kwargs):
        self.manage_outliers()
        self.format_target()
        self.features_engineering()

    def manage_outliers(self):
        pass

    def format_target(self):
        pass

    def features_engineering(self):
        self.train = self.train[self.train.GrLivArea < 4500]
        self.test_ids = self.test['Id']
        self.train.drop(['Id'], axis=1, inplace=True)
        self.test.drop(['Id'], axis=1, inplace=True)

        print("Train set size:", self.train.shape)
        print("Test set size:", self.test.shape)

        self.train.SalePrice = np.log1p(self.train.SalePrice)

        y = self.train.SalePrice.reset_index(drop=True)
        train_features = self.train.drop(['SalePrice'], axis=1)
        test_features = self.test

        features = pd.concat([train_features, test_features]).reset_index(
            drop=True)
        print(features.shape)

        nulls = np.sum(features.isnull())
        nullcols = nulls.loc[(nulls != 0)]
        dtypes = features.dtypes
        dtypes2 = dtypes.loc[(nulls != 0)]
        info = pd.concat([nullcols, dtypes2], axis=1).sort_values(by=0,
                                                                  ascending=False)
        print(info)
        print("There are", len(nullcols), "columns with missing values")

        features['Functional'] = features['Functional'].fillna('Typ')
        features['Electrical'] = features['Electrical'].fillna("SBrkr")
        features['KitchenQual'] = features['KitchenQual'].fillna("TA")

        features['Exterior1st'] = features['Exterior1st'].fillna(
            features['Exterior1st'].mode()[0])
        features['Exterior2nd'] = features['Exterior2nd'].fillna(
            features['Exterior2nd'].mode()[0])

        features['SaleType'] = features['SaleType'].fillna(
            features['SaleType'].mode()[0])

        pd.set_option('max_columns', None)
        print(features[features['PoolArea'] > 0 & features['PoolQC'].isnull()])

        features.loc[2418, 'PoolQC'] = 'Fa'
        features.loc[2501, 'PoolQC'] = 'Gd'
        features.loc[2597, 'PoolQC'] = 'Fa'

        features.loc[2124, 'GarageYrBlt'] = features['GarageYrBlt'].median()
        features.loc[2574, 'GarageYrBlt'] = features['GarageYrBlt'].median()

        features.loc[2124, 'GarageFinish'] = features['GarageFinish'].mode()[0]
        features.loc[2574, 'GarageFinish'] = features['GarageFinish'].mode()[0]

        features.loc[2574, 'GarageCars'] = features['GarageCars'].median()

        features.loc[2124, 'GarageArea'] = features['GarageArea'].median()
        features.loc[2574, 'GarageArea'] = features['GarageArea'].median()

        features.loc[2124, 'GarageQual'] = features['GarageQual'].mode()[0]
        features.loc[2574, 'GarageQual'] = features['GarageQual'].mode()[0]

        features.loc[2124, 'GarageCond'] = features['GarageCond'].mode()[0]
        features.loc[2574, 'GarageCond'] = features['GarageCond'].mode()[0]

        basement_columns = ['BsmtQual', 'BsmtCond', 'BsmtExposure',
                            'BsmtFinType1',
                            'BsmtFinType2', 'BsmtFinSF1', 'BsmtFinSF2',
                            'BsmtUnfSF',
                            'TotalBsmtSF']

        tempdf = features[basement_columns]
        tempdfnulls = tempdf[tempdf.isnull().any(axis=1)]

        features.loc[332, 'BsmtFinType2'] = 'ALQ'  # since smaller than SF1
        features.loc[947, 'BsmtExposure'] = 'No'
        features.loc[1485, 'BsmtExposure'] = 'No'
        features.loc[2038, 'BsmtCond'] = 'TA'
        features.loc[2183, 'BsmtCond'] = 'TA'
        features.loc[
            2215, 'BsmtQual'] = 'Po'  # v small basement so let's do Poor.
        features.loc[2216, 'BsmtQual'] = 'Fa'  # similar but a bit bigger.
        features.loc[
            2346, 'BsmtExposure'] = 'No'  # unfinished bsmt so prob not.
        features.loc[2522, 'BsmtCond'] = 'Gd'  # cause ALQ for bsmtfintype1

        subclass_group = features.groupby('MSSubClass')
        Zoning_modes = subclass_group['MSZoning'].apply(lambda x: x.mode()[0])

        features['MSZoning'] = features.groupby('MSSubClass')[
            'MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

        objects = []
        for i in features.columns:
            if features[i].dtype == object:
                objects.append(i)

        features.update(features[objects].fillna('None'))

        nulls = np.sum(features.isnull())
        nullcols = nulls.loc[(nulls != 0)]
        dtypes = features.dtypes
        dtypes2 = dtypes.loc[(nulls != 0)]
        info = pd.concat([nullcols, dtypes2], axis=1).sort_values(by=0,
                                                                  ascending=False)
        print(info)
        print("There are", len(nullcols), "columns with missing values")

        neighborhood_group = features.groupby('Neighborhood')
        lot_medians = neighborhood_group['LotFrontage'].median()

        features['LotFrontage'] = features.groupby('Neighborhood')[
            'LotFrontage'].transform(lambda x: x.fillna(x.median()))
        # Filling in the rest of the NA's

        numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32',
                          'float64']
        numerics = []
        for i in features.columns:
            if features[i].dtype in numeric_dtypes:
                numerics.append(i)

        features.update(features[numerics].fillna(0))

        nulls = np.sum(features.isnull())
        nullcols = nulls.loc[(nulls != 0)]
        dtypes = features.dtypes
        dtypes2 = dtypes.loc[(nulls != 0)]
        info = pd.concat([nullcols, dtypes2], axis=1).sort_values(by=0,
                                                                  ascending=False)
        print(info)
        print("There are", len(nullcols), "columns with missing values")

        features.loc[2590, 'GarageYrBlt'] = 2007

        # factors = ['MSSubClass', 'MoSold']
        factors = ['MSSubClass']

        for i in factors:
            features.update(features[i].astype('str'))

        from scipy.stats import skew

        numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32',
                          'float64']
        numerics2 = []
        for i in features.columns:
            if features[i].dtype in numeric_dtypes:
                numerics2.append(i)

        skew_features = features[numerics2].apply(
            lambda x: skew(x)).sort_values(ascending=False)
        skews = pd.DataFrame({'skew': skew_features})

        from scipy.special import boxcox1p
        from scipy.stats import boxcox_normmax

        high_skew = skew_features[skew_features > 0.5]
        high_skew = high_skew
        skew_index = high_skew.index

        for i in skew_index:
            features[i] = boxcox1p(features[i],
                                   boxcox_normmax(features[i] + 1))

        skew_features2 = features[numerics2].apply(
            lambda x: skew(x)).sort_values(ascending=False)
        skews2 = pd.DataFrame({'skew': skew_features2})

        objects3 = []
        for i in features.columns:
            if features[i].dtype == object:
                objects3.append(i)

        print("Training Set incomplete cases")

        sums_features = features[objects3].apply(lambda x: len(np.unique(x)))
        sums_features.sort_values(ascending=False)

        print(features['Street'].value_counts())
        print('-----')
        print(features['Utilities'].value_counts())
        print('-----')
        print(features['CentralAir'].value_counts())
        print('-----')
        print(features['PavedDrive'].value_counts())

        features = features.drop(['Utilities', 'Street'], axis=1)

        features['Total_sqr_footage'] = (
                    features['BsmtFinSF1'] + features['BsmtFinSF2'] +
                    features['1stFlrSF'] + features['2ndFlrSF'])

        features['Total_Bathrooms'] = (
                    features['FullBath'] + (0.5 * features['HalfBath']) +
                    features['BsmtFullBath'] + (
                                0.5 * features['BsmtHalfBath']))

        features['Total_porch_sf'] = (
                    features['OpenPorchSF'] + features['3SsnPorch'] +
                    features['EnclosedPorch'] + features['ScreenPorch'] +
                    features['WoodDeckSF'])

        # simplified features
        features['haspool'] = features['PoolArea'].apply(
            lambda x: 1 if x > 0 else 0)
        features['has2ndfloor'] = features['2ndFlrSF'].apply(
            lambda x: 1 if x > 0 else 0)
        features['hasgarage'] = features['GarageArea'].apply(
            lambda x: 1 if x > 0 else 0)
        features['hasbsmt'] = features['TotalBsmtSF'].apply(
            lambda x: 1 if x > 0 else 0)
        features['hasfireplace'] = features['Fireplaces'].apply(
            lambda x: 1 if x > 0 else 0)

        print(features.shape)

        final_features = pd.get_dummies(features).reset_index(drop=True)

        X = final_features.iloc[:len(y), :]
        testing_features = final_features.iloc[len(X):, :]

        print(X.shape)
        print(testing_features.shape)

        # import statsmodels.api as sm
        # ols = sm.OLS(endog = y, exog = X)
        # fit = ols.fit()
        # test2 = fit.outlier_test()['bonf(p)']
        # outliers = list(test2[test2<1e-3].index)

        outliers = [30, 88, 462, 631, 1322]

        X = X.drop(X.index[outliers])
        y = y.drop(y.index[outliers])

        overfit = []
        for i in X.columns:
            counts = X[i].value_counts()
            zeros = counts.iloc[0]
            if zeros / len(X) * 100 > 99.94:
                overfit.append(i)

        overfit = list(overfit)

        overfit.append('MSZoning_C (all)')
        print(overfit)

        X.drop(overfit, axis=1, inplace=True)
        testing_features.drop(overfit, axis=1, inplace=True)

        self.train = X
        self.y_train = y
        self.test = testing_features


