import json
import argparse
import math
import numpy as np
from sklearn import metrics


def run(test_location):
    y_test = np.load(f"{test_location}/y_test.npy")
    y_pred_proba = np.load(f"{test_location}/y_pred_proba.npy")
    # Probability of positive class
    y_pred = y_pred_proba[:, 1]
    print(len(y_pred))
    print(y_pred)
    print(len(y_test))
    precision, recall, prc_thresholds = metrics.precision_recall_curve(y_test, y_pred)
    print(len(precision))
    avg_prec = metrics.average_precision_score(y_test, y_pred)
    roc_auc = metrics.roc_auc_score(y_test, y_pred)

    with open("metrics.json", "w") as fd:
        json.dump({"avg_prec": avg_prec, "roc_auc": roc_auc}, fd, indent=4)

    prc_points = list(zip(precision, recall, prc_thresholds))

    with open("prc.json", "w") as outfile:
        pr_curve = {
                "prc": [
                    {"precision": p, "recall": r, "threshold": float(t)}
                    for p, r, t in prc_points
                ]
            }
        json.dump(pr_curve, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_location", action="store", dest="test_location", type=str, required=True, help="Location of test data")
    args = parser.parse_args()
    run(args.test_location)
