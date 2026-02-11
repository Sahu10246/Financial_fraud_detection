DATA_PATH = "fraud.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 12,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}
