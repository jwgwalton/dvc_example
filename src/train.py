import os
import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def run(train_location, test_location):
    X_train = np.load(f"{train_location}/X_train.npy")
    y_train = np.load(f"{train_location}/y_train.npy")

    X_test = np.load(f"{test_location}/X_test.npy")

    model = RandomForestClassifier(n_estimators=40)
    model.fit(X_train, y_train)
    predict_proba = model.predict_proba(X_test)

    os.makedirs(test_location, exist_ok=True)
    np.save(f"{test_location}/y_pred_proba.npy", predict_proba)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_location", action="store", dest="train_location", type=str, required=True, help="Location of train data")
    parser.add_argument("--test_location", action="store", dest="test_location", type=str, required=True, help="Location of test data")
    args = parser.parse_args()
    run(args.train_location, args.test_location)
