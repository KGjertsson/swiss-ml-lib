from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from scipy.special import boxcox1p
from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder

from . import abstract_data_cleaner


class Top4PercentDataCleaner(abstract_data_cleaner.AbstractDataCleaner):
    def __init__(self, data_root_dir, verbose):
        super().__init__(data_root_dir, verbose)
        t_0 = time()
        if self.verbose:
            print('Loading data')
        self.train = pd.read_csv(Path(self.data_root_dir, 'train.csv'))
        self.test = pd.read_csv(Path(self.data_root_dir, 'test.csv'))

        self.test_ids = self.test['Id']
        self.train.drop("Id", axis=1, inplace=True)
        self.test.drop("Id", axis=1, inplace=True)

        if self.verbose:
            print('Done ({} s)'.format(time() - t_0))

    def __call__(self):
        self.format_target()
        self.manage_outliers()
        self.features_engineering()

    def format_target(self):
        self.train["SalePrice"] = np.log1p(self.train["SalePrice"])

    def manage_outliers(self):
        self.train = self.train.drop(
            self.train[(self.train['GrLivArea'] > 4000) &
                       (self.train['SalePrice'] < 300000)].index)

    def features_engineering(self):
        ntrain = self.train.shape[0]
        self.y_train = self.train.SalePrice.values
        all_data = pd.concat(
            (self.train, self.test), sort=True).reset_index(drop=True)
        all_data.drop(['SalePrice'], axis=1, inplace=True)

        # missing data
        all_data = self._input_missing_data(all_data)

        # transform some numerical values to categorical
        all_data = self._transform_numerical_to_categorical(all_data)

        # Adding total sqfootage feature
        all_data['TotalSF'] = \
            all_data['TotalBsmtSF'] \
            + all_data['1stFlrSF'] \
            + all_data['2ndFlrSF']

        # adjust skewed parameters
        all_data = self._adjust_skewed_parameters(all_data)

        self.train = all_data[:ntrain]
        self.test = all_data[ntrain:]

    def _input_missing_data(self, all_data):
        # Imputing missing values

        # PoolQC : data description says NA means "No Pool". That make sense,
        # given the huge ratio of missing value (+99%) and majority of houses
        # have no Pool at all in general.
        all_data["PoolQC"] = all_data["PoolQC"].fillna("None")

        # MiscFeature : data description says NA means "no misc feature"
        all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")

        # Alley : data description says NA means "no alley access"
        all_data["Alley"] = all_data["Alley"].fillna("None")

        # Fence : data description says NA means "no fence"
        all_data["Fence"] = all_data["Fence"].fillna("None")

        # FireplaceQu : data description says NA means "no fireplace"
        all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

        # LotFrontage : Since the area of each street connected to the house
        # property most likely # have a similar area to other houses in its
        # neighborhood , we can fill in missing values by the median
        # LotFrontage of the neighborhood. Group by neighborhood and fill in
        # missing value by the median LotFrontage of all the neighborhood
        all_data["LotFrontage"] = \
            all_data.groupby("Neighborhood")["LotFrontage"].transform(
                lambda x: x.fillna(x.median()))

        # GarageType, GarageFinish, GarageQual and GarageCond : Replacing
        # missing data with None
        for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
            all_data[col] = all_data[col].fillna('None')

        # GarageYrBlt, GarageArea and GarageCars : Replacing missing data with
        # 0 (Since No garage = no cars in such garage.)
        for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
            all_data[col] = all_data[col].fillna(0)

        # BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and
        # BsmtHalfBath : missing values are # likely zero for having no
        # basement
        for col in (
                'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                'BsmtFullBath', 'BsmtHalfBath'):
            all_data[col] = all_data[col].fillna(0)

        # BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2 :
        # For all these categorical basement-related features, NaN means that
        # there is no basement.
        for col in (
                'BsmtQual', 'BsmtCond', 'BsmtExposure',
                'BsmtFinType1', 'BsmtFinType2'):
            all_data[col] = all_data[col].fillna('None')

        # MasVnrArea and MasVnrType : NA most likely means no masonry veneer
        # for these houses. We can fill 0 for the area and None for the type.
        all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
        all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

        # MSZoning (The general zoning classification) : 'RL' is by far the
        # most common value. So we can fill in missing values with 'RL'
        all_data['MSZoning'] = \
            all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

        # Utilities : For this categorical feature all records are "AllPub",
        # except for one "NoSeWa" and 2 NA . Since the house with 'NoSewa' is
        # in the training set, this feature won't help in predictive modelling.
        # We can then safely remove it.
        all_data = all_data.drop(['Utilities'], axis=1)

        # Functional : data description says NA means typical
        all_data["Functional"] = all_data["Functional"].fillna("Typ")

        # Electrical : It has one NA value. Since this feature has mostly
        # 'SBrkr', we can set that for the missing value.
        all_data['Electrical'] = \
            all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

        # KitchenQual: Only one NA value, and same as Electrical, we set 'TA'
        # (which is the most frequent) for the missing value in KitchenQual.
        all_data['KitchenQual'] = \
            all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

        # Exterior1st and Exterior2nd : Again Both Exterior 1 & 2 have only one
        # missing value. We will just substitute in the most common string
        all_data['Exterior1st'] = \
            all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
        all_data['Exterior2nd'] = \
            all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

        # SaleType : Fill in again with most frequent which is "WD"
        all_data['SaleType'] = \
            all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

        # MSSubClass : Na most likely means No building class. We can replace
        # missing values with None
        all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

        return all_data

    def _transform_numerical_to_categorical(self, all_data):
        # MSSubClass=The building class
        all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

        # Changing OverallCond into a categorical variable
        all_data['OverallCond'] = all_data['OverallCond'].astype(str)

        # Year and month sold are transformed into categorical features.
        all_data['YrSold'] = all_data['YrSold'].astype(str)
        all_data['MoSold'] = all_data['MoSold'].astype(str)

        # Label Encoding some categorical variables that may contain
        # information in their ordering set
        cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual',
                'GarageCond', 'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC',
                'KitchenQual', 'BsmtFinType1', 'BsmtFinType2', 'Functional',
                'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
                'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir',
                'MSSubClass', 'OverallCond', 'YrSold', 'MoSold')

        # process columns, apply LabelEncoder to categorical features
        for c in cols:
            lbl = LabelEncoder()
            lbl.fit(list(all_data[c].values))
            all_data[c] = lbl.transform(list(all_data[c].values))

        return all_data

    def _adjust_skewed_parameters(self, all_data):
        numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

        # Check the skew of all numerical features
        skewed_feats = all_data[numeric_feats].apply(
            lambda x: skew(x.dropna())).sort_values(
            ascending=False)
        skewness = pd.DataFrame({'Skew': skewed_feats})

        # Box Cox Transformation of (highly) skewed features
        skewness = skewness[abs(skewness) > 0.75]

        skewed_features = skewness.index
        lam = 0.15
        for feat in skewed_features:
            all_data[feat] = boxcox1p(all_data[feat], lam)

        # Getting dummy categorical features
        all_data = pd.get_dummies(all_data)

        return all_data
