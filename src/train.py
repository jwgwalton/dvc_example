import os
import argparse
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier


def run(train_location, test_location, params):
    X_train = np.load(f"{train_location}/X_train.npy")
    y_train = np.load(f"{train_location}/y_train.npy")

    X_test = np.load(f"{test_location}/X_test.npy")

    n_estimators = params["n_estimators"]
    min_split = params["min_split"]
    model = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_split)
    model.fit(X_train, y_train)
    predict_proba = model.predict_proba(X_test)

    os.makedirs(test_location, exist_ok=True)
    np.save(f"{test_location}/y_pred_proba.npy", predict_proba)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_location", action="store", dest="train_location", type=str, required=True, help="Location of train data")
    parser.add_argument("--test_location", action="store", dest="test_location", type=str, required=True, help="Location of test data")
    args = parser.parse_args()
    params = yaml.safe_load(open("params.yaml"))["train"]
    run(args.train_location, args.test_location, params)
