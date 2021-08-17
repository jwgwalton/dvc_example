import json
import yaml
import numpy as np
from sklearn import metrics


def run(data_location):
    y_test = np.load(f"{data_location['test_data_location']}/y_test.npy")
    y_pred_proba = np.load(f"{data_location['test_data_location']}/y_pred_proba.npy")
    # Probability of positive class
    y_pred = y_pred_proba[:, 1]

    precision, recall, prc_thresholds = metrics.precision_recall_curve(y_test, y_pred)
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
    data_location = yaml.safe_load(open("params.yaml"))["data_location"]
    run(data_location)
