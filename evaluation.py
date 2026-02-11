from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix
)
import pandas as pd


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nROC-AUC Score:", roc_auc_score(y_test, y_prob))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))


def feature_importance(model, feature_names):

    importance = pd.Series(
        model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)

    print("\nTop 10 Important Features:\n")
    print(importance.head(10))
