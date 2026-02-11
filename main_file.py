from config import DATA_PATH, TEST_SIZE, RANDOM_STATE, RF_PARAMS
from data_processing import load_data, clean_data
from feature_engineering import create_features
from model import train_model
from evaluation import evaluate_model, feature_importance
from bussiness import print_business_summary


def main():

    df = load_data(DATA_PATH)

    df = clean_data(df)

    df = create_features(df)

    model, X_test, y_test, feature_names = train_model(
        df,
        TEST_SIZE,
        RANDOM_STATE,
        RF_PARAMS
    )

    evaluate_model(model, X_test, y_test)

    feature_importance(model, feature_names)

    print_business_summary()


if __name__ == "__main__":
    main()
