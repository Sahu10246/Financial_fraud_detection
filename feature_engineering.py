def create_features(df):

    df["balanceDiffOrig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balanceDiffDest"] = df["oldbalanceDest"] - df["newbalanceDest"]

    df["isLargeTransaction"] = (
        df["amount"] > df["amount"].quantile(0.95)
    ).astype(int)

    return df
