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


class ColumnTransformerSetDate(BaseEstimator, TransformerMixin):

    def __init__(self, d_encode, d_decode, h_encode, h_decode):
        self.columns = 'date'
        self.h_encode = h_encode
        self.h_decode = h_decode
        self.d_encode = d_encode
        self.d_decode = d_decode

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: np.ndarray):
        X = pd.DataFrame(X, columns=self.d_decode)
        X['date'] = pd.to_datetime(X['date'])
        X = X.set_index('date', drop=True, inplace=False)
        X.sort_index(inplace=True)
        return X
