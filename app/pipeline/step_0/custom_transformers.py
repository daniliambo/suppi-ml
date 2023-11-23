import os
from collections import defaultdict
import pprint


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import pipeline
from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

import os
from collections import defaultdict
import pprint

np.set_printoptions(suppress=True)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pprint.PrettyPrinter(indent=4)


class ColumnTransformerDrop(BaseEstimator, TransformerMixin):
    """
    Drops columns from DataFrame.
    Doesn't work with np.ndarray.
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame):
        X.drop(columns=self.columns, axis=1, inplace=True)
        return X


class ColumnTransformerDropNaBase(BaseEstimator, TransformerMixin):
    """
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X):
        X.dropna(subset=self.columns, inplace=True)
        return X


class ColumnTransformerJoinNonInformative(BaseEstimator, TransformerMixin):
    """
    Joins NonInformative columns.
    Doesn't work with np.ndarray.
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def _get_present_columns(self, X, cols):
        f = set
        output = set()
        for col in cols:
            if col in X.columns:
                output.add(col)
        return list(output)

    def _extend_numeric_columns(self, X, cols):
        f = set
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

        numeric_columns = X[cols].select_dtypes(include=numerics).columns

        cols_without_numerics = f(cols) - f(numeric_columns)

        for col in cols_without_numerics:
            if X[col].str.isnumeric().sum() == X[col].size:
                X[col].astype(float, inplace=True)
                numeric_columns.append(col)
        return numeric_columns

    def transform(self, X: pd.DataFrame):
        cols = self.columns
        cols = self._get_present_columns(X, cols)
        numeric_columns = self._extend_numeric_columns(X, cols)

        # additional cols
        X['sum_' + '_'.join(numeric_columns)] = X.loc[:,
                                                      numeric_columns].sum(axis=1)
        ao, ar = 'accruals_for_orders', 'accruals_for_return'

        # additional cols
        X['accruals'] = X[ao] + X[ar]

        columns_to_drop = list(numeric_columns)
        columns_to_drop.extend([ao, ar])
        X.drop(columns_to_drop, axis=1, inplace=True)
        return X


class ColumnTransformerDropNonInformative(BaseEstimator, TransformerMixin):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.base = ['shop_id', 'offer_id', 'date']

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X):
        f = set
        cols_with_zeros = X.astype(bool).sum(axis=0)
        cols_with_zeros = cols_with_zeros[cols_with_zeros == 0].index.tolist()

        cols_with_single_unique = X.columns[(
            X.apply(pd.Series.nunique) == 1).values].tolist()

        cols_with_single_unique = list(
            f(cols_with_single_unique) - f(self.base + ['delivery_schema']))
        X.drop(cols_with_zeros +
               cols_with_single_unique, axis=1, inplace=True)
        return X


class ColumnTransformerFillNaData(BaseEstimator, TransformerMixin):

    """
    Handles columns:

    ['warehouse_name',
    'delivery_schema',
    'gds_profitability_after_tax',
    'shop_profitability_after_tax']
    """

    def __init__(self, columns):
        self.columns = columns
        self.col_group_by_1 = 'shop_id'
        self.col_group_by_2 = 'offer_id'

    def _fill_mode(self, X: pd.DataFrame, col: str):
        # X[col].fillna(X.groupby([self.col_group_by_1, self.col_group_by_2])[
        #               col].transform(lambda x: pd.Series.mode(x).iat[0]), inplace=True)

        # X[col].fillna(X.groupby(self.col_group_by_1)[col].transform(
        #     lambda x: pd.Series.mode(x).iat[0]), inplace=True)

        mode_ = pd.Series.mode(X[col]).iat[0]
        X[col].fillna(mode_, inplace=True)

    def _fill_mean(self, X: pd.DataFrame, col: str):
        X[col].fillna(X.groupby([self.col_group_by_1, self.col_group_by_2])[
                      col].transform(lambda x: pd.Series.mean(x)), inplace=True)

        X[col].fillna(X.groupby(self.col_group_by_1)[col].transform(
            lambda x: pd.Series.mean(x)), inplace=True)

        X[col].fillna(pd.Series.mean(X[col]), inplace=True)

    def _fill_warehouse_name(self, X: pd.DataFrame, col: str):
        self._fill_mode(X, col)

    def _fill_delivery_schema(self, X: pd.DataFrame, col: str):
        self._fill_mode(X, col)

    def _fill_gds_profitability_after_tax(self, X: pd.DataFrame, col: str):
        self._fill_mean(X, col)

    def _fill_shop_profitability_after_tax(self, X: pd.DataFrame, col: str):
        self._fill_mean(X, col)

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame):

        # additional dropna
        X = X.dropna(subset=['promotion_reviews_prod', 'acquiring_cost_prod', 'gds_net_profit_after_tax', 'gds_net_profit_bfr_tax',
                     'placement_cost_prod', 'other_cost_prod', 'cross_docking_cost_prod', 'promotion_advertising_prod'], axis=0, inplace=False, how='any')

        is_na = X.isna().sum()

        na_cols = is_na.loc[is_na > 0].index
        diff_col = set(na_cols).difference(set(self.columns))

        assert (len(diff_col) == 0)

        for col in self.columns:
            if col in na_cols:
                getattr(ColumnTransformerFillNaData,
                        '_fill_' + col)(self, X, col)
        return X


class ColumnTransformerGroupBy(BaseEstimator, TransformerMixin):
    """
    """

    def __init__(self, group_by_columns):
        self.columns = group_by_columns
        self.aggregate_map = {}

    def _assemble_map(self, X):
        aggregate_map = dict()

        def handler_mode(x):
            return pd.Series.mode(x).iat[0]

        n_cols = len(X.columns)
        non_numeric = dict(
            zip(X.select_dtypes(include='object').columns, n_cols * [handler_mode]))
        numeric = dict(
            zip(X.select_dtypes(exclude='object').columns, n_cols * [pd.Series.sum]))

        aggregate_map.update(non_numeric)
        aggregate_map.update(numeric)

        for el in self.columns:
            aggregate_map.pop(el)
        return aggregate_map

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame):
        f = set
        self.aggregate_map = self._assemble_map(X)
        assert (f(self.aggregate_map.keys()) == f(X.columns) - f(self.columns))

        X = X.groupby(self.columns).agg(self.aggregate_map)

        assert (np.where(X.groupby(self.columns).count()
                > 1, 1, 0).sum().sum() == 0)
        X = X.reset_index()
        return X


class ColumnTransformerOrdinalEncoding(BaseEstimator, TransformerMixin):
    """
    """

    def __init__(self, columns):
        self.columns = columns
        self.ordinal_encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=-1)

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self.ordinal_encoder.fit(X.loc[:, self.columns])
        return self

    def transform(self, X: pd.DataFrame):
        X.loc[:, self.columns] = self.ordinal_encoder.transform(
            X.loc[:, self.columns])
        return X
