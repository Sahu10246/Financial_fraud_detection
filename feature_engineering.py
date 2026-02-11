def create_features(df):
    

    """
    Create additional features such as balance differences 
    and large transaction indicator for fraud detection.
    """
    
    df["balanceDiffOrig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balanceDiffDest"] = df["oldbalanceDest"] - df["newbalanceDest"]

    df["isLargeTransaction"] = (
        df["amount"] > df["amount"].quantile(0.95)
    ).astype(int)

    return df
