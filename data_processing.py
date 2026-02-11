import pandas as pd
import numpy as np


def load_data(path):
    df = pd.read_csv(path)
    print("Dataset Shape:", df.shape)
    return df


def clean_data(df):

    df = df.drop_duplicates()
    df = df.fillna(0)

    # Outlier handling (IQR)
    Q1 = df["amount"].quantile(0.25)
    Q3 = df["amount"].quantile(0.75)
    IQR = Q3 - Q1

    df = df[
        (df["amount"] >= Q1 - 1.5 * IQR) &
        (df["amount"] <= Q3 + 1.5 * IQR)
    ]

    # Remove multicollinearity from the data 
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    drop_cols = [
        col for col in upper.columns
        if any(upper[col] > 0.90)
    ]

    df = df.drop(columns=drop_cols)

    print("After Cleaning:", df.shape)

    return df
