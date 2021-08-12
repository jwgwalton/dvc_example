import os
import argparse
import numpy as np
from sklearn.datasets import load_breast_cancer


def run(data_location):
    """
    Initial data creation code. When data is changed it will be versioned with DVC and pushed to remote storage for checkout.
    :return:
    """
    X, y = load_breast_cancer(return_X_y=True)

    os.makedirs(data_location, exist_ok=True)

    np.save(f"{data_location}/X", X)
    np.save(f"{data_location}/y", y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_location", action="store", dest="data_location", type=str, required=True, help="Location of data")
    args = parser.parse_args()
    run(args.data_location)
