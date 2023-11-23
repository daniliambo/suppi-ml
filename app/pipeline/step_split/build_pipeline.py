from pipeline.step_2.build_pipeline import run_pipeline, plot
import os
from collections import defaultdict
import pprint


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


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
import pprint

from utils.reader import ReadData


def compose_pipeline():
    reader = ReadData()
    df_meta = reader.read('meta')
    df_feature = reader.read('feature')
    return df_meta


def run_pipeline(n):
    df = compose_pipeline()
    return df, bundles


def load_and_preprocess():
    df, bundles = run_pipeline()
    return bundles


if __name__ == "__main__":
    bundles = load_and_preprocess()
