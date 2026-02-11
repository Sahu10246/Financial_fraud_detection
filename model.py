from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def train_model(df, test_size, random_state, params):

    X = df.drop(columns=["isFraud"])
    y = df["isFraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    return model, X_test, y_test, X.columns
