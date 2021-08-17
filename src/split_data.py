import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split


def run(data_location, params):
    """
    Split data into train and test sets
    """
    print(data_location)
    print(data_location['raw_data_location'])
    X = np.load(f"{data_location['raw_data_location']}/X.npy")
    y = np.load(f"{data_location['raw_data_location']}/y.npy")

    random_state = params["random_state"]
    test_size = params["test_size"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_size)

    os.makedirs(data_location['train_data_location'], exist_ok=True)
    os.makedirs(data_location['test_data_location'], exist_ok=True)

    np.save(f"{data_location['train_data_location']}/X_train", X_train)
    np.save(f"{data_location['train_data_location']}/y_train", y_train)

    np.save(f"{data_location['test_data_location']}/X_test", X_test)
    np.save(f"{data_location['test_data_location']}/y_test", y_test)


if __name__ == '__main__':
    data_location = yaml.safe_load(open("params.yaml"))["data_location"]
    params = yaml.safe_load(open("params.yaml"))["split_data"]
    run(data_location, params)
