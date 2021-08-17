import os
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier


def run(data_location, params):
    X_train = np.load(f"{data_location['train_data_location']}/X_train.npy")
    y_train = np.load(f"{data_location['train_data_location']}/y_train.npy")
    X_test = np.load(f"{data_location['test_data_location']}/X_test.npy")

    n_estimators = params["n_estimators"]
    min_split = params["min_split"]

    model = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_split)
    model.fit(X_train, y_train)
    predict_proba = model.predict_proba(X_test)

    os.makedirs(data_location['test_data_location'], exist_ok=True)
    np.save(f"{data_location['test_data_location']}/y_pred_proba.npy", predict_proba)


if __name__ == '__main__':
    data_location = yaml.safe_load(open("params.yaml"))["data_location"]
    params = yaml.safe_load(open("params.yaml"))["train"]
    run(data_location, params)
