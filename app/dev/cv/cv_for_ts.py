# cv_split for time-series

from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import os
import pickle
import pandas as pd


def preprocessing(df):
    TARGET = 'TARGET'

    def set_target(df):
        qp, qr = 'quantity_of_paid', 'quantity_of_return',
        df[TARGET] = df[qp] - df[qr]
        df.drop([qp, qr], axis=1, inplace=True)
        return df

    def set_datetime(df):
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date', drop=True, inplace=False)
        df.sort_index(inplace=True)
        return df

    def get_split(df):
        test_size = 0.8
        n = df.shape[0]
        full_idx = pd.date_range(
            start=df.index.min(), end=df.index.max(), freq='0.2T')
        print(full_idx)
        df_train = df.loc[:int(n * test_size), ]
        df_test = df.loc[int(n * test_size):, ]
        return df_train, df_test

    df = set_target(df)
    df = set_datetime(df)
    df_train, df_test = get_split(df)

    return df_train, df_test

def get_splits(path_to_save: str, df: pd.DataFrame):
    TARGET = 'TARGET'

    df_train, df_test = preprocessing(df)
    test_size = df_test.shape[0]

    if os.path.exists(path_to_save):
        with open(path_to_save, 'rb') as f:
            indices = pickle.load(f)
    else:
        indices = {}
        indices['train_indices'], indices['test_indices'] = train_test_split(
            np.arange(len(df_train)),
            test_size=test_size,
            stratify=df_train[TARGET],
            shuffle=False,
            random_state=42
        )

        df_train = df_train.iloc[indices['train_indices']]
        cv_splitter = StratifiedKFold(
            n_splits=5,
            shuffle=False,
            random_state=42
        )

        indices['cv_iterable'] = []
        for train_indices, val_indices in cv_splitter.split(df_train.drop(TARGET, axis=1), df_train[TARGET]):
            indices['cv_iterable'].append(
                (train_indices, val_indices)
            )

        with open(path_to_save, 'wb+') as f:
            pickle.dump(indices, f)

    with open(path_to_save, 'rb') as f:
        indices = pickle.load(f)
