import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split


def run(raw_data_location, train_location, test_location):
    """
    Split data into train and test sets
    """

    X = np.load(f"{raw_data_location}/X.npy")
    y = np.load(f"{raw_data_location}/y.npy")

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    os.makedirs(train_location, exist_ok=True)
    os.makedirs(test_location, exist_ok=True)

    np.save(f"{train_location}/X_train", X_train)
    np.save(f"{train_location}/y_train", y_train)

    np.save(f"{test_location}/X_test", X_test)
    np.save(f"{test_location}/y_test", y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_location", action="store", dest="raw_data_location", type=str, required=True, help="Location of raw data")
    parser.add_argument("--train_location", action="store", dest="train_location", type=str, required=True, help="Location of training data")
    parser.add_argument("--test_location", action="store", dest="test_location", type=str, required=True, help="Location of test data")
    args = parser.parse_args()
    run(args.raw_data_location, args.train_location, args.test_location)
