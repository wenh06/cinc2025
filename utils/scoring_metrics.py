from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch_ecg.components.metrics import ClassificationMetrics
from torch_ecg.utils.misc import make_serializable

from helper_code import compute_accuracy, compute_auc, compute_confusion_matrix, compute_f_measure
from outputs import CINC2025Outputs

__all__ = [
    "compute_challenge_metrics",
]


def compute_challenge_metrics(
    labels: Sequence[Dict[str, Union[np.ndarray, torch.Tensor, List[dict]]]],
    outputs: Sequence[CINC2025Outputs],
    keeps: Optional[Union[str, Sequence[str]]] = None,
) -> Dict[str, float]:
    """Compute the challenge metrics.

    Parameters
    ----------
    labels : Sequence[Dict[str, Union[np.ndarray, torch.Tensor, List[dict]]]]
        The labels for the records.
        `labels` is produced by the dataset class (ref. dataset.py).
    outputs : Sequence[CINC2025Outputs]
        The outputs for the records.
    keeps : Union[str, Sequence[str]], optional
        Metrics to keep, available options are "chagas", "arr_diag".
        By default all metrics are computed.

    Returns
    -------
    Dict[str, float]
        The computed challenge metrics for "chagas", "arr_diag" (at least one of them).
        nan values are returned for the metrics that are not computed due to missing outputs.

    Examples
    --------
    >>> labels = [
    ...     {
    ...         "arr_diag": np.array([[0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0, 0, 0]]),
    ...         "chagas": np.array([0, 1]),
    ...     }
    ... ]
    >>> outputs = [
    ...     CINC2025Outputs(
    ...         chagas=[0, 0],
    ...         chagas_prob=np.array([[0.9, 0.1], [0.7, 0.3]]),
    ...         arr_diag=[["RBBB", "AF"], ["SB", "ST"]],
    ...         arr_diag_prob=np.array([
    ...             [0.1, 0.9, 0.1, 0.1, 0.1, 0.7, 0.1, 0.1],
    ...             [0.3, 0.1, 0.1, 0.6, 0.7, 0.1, 0.1, 0.1],
    ...         ]),
    ...         arr_diag_classes=["1dAVb", "RBBB", "LBBB", "SB", "ST", "AF", "NORM", "OTHER"],
    ...     )
    ... ]
    >>> compute_challenge_metrics(labels, outputs)

    """
    metrics = {}
    if keeps is None:
        # keeps = ["chagas", "arr_diag"]
        keeps = ["chagas"]
    elif isinstance(keeps, str):
        keeps = [keeps]
    keeps = [keep.lower() for keep in keeps]
    if "chagas" in keeps:
        metrics.update({f"chagas_{metric}": value for metric, value in compute_chagas_metrics(labels, outputs).items()})
        metrics["challenge_score"] = metrics.pop("chagas_challenge_score")
    if "arr_diag" in keeps:
        metrics.update({f"arr_diag_{metric}": value for metric, value in compute_arr_diag_metrics(labels, outputs).items()})
    return metrics


def compute_chagas_metrics(
    labels: Sequence[Dict[str, Union[np.ndarray, torch.Tensor, List[dict]]]],
    outputs: Sequence[CINC2025Outputs],
) -> Dict[str, float]:
    """Compute the metrics for the "chagas" prediction (binary classification) task.

    Parameters
    ----------
    labels : Sequence[Dict[str, Union[np.ndarray, torch.Tensor, List[dict]]]]
        The labels for the records, containing the "chagas" field.
        The "chagas" field is a 1D array of shape `(num_samples,)` with binary values,
        or a 2D array of shape `(num_samples, num_classes)` with probabilities (0 or 1).
    outputs : Sequence[CINC2025Outputs]
        The outputs for the records, containing the "chagas" field.
        The "chagas" field is a 1D array of shape `(num_samples,)` with binary values,
        or a 2D array of shape `(num_samples, num_samples)` with probabilities (0 to 1).

    Returns
    -------
    Dict[str, float]
        The computed challenge metrics for the "chagas" prediction task.

    Examples
    --------
    >>> labels = [{"chagas": np.array([0, 1, 0])}]
    >>> outputs = [CINC2025Outputs(chagas=np.array([0, 1, 1]), chagas_prob=np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7]]))]
    >>> compute_chagas_metrics(labels, outputs)
    {'challenge_score': 0.0, 'auroc': 1.0, 'auprc': 1.0, 'accuracy': 0.6666666666666666, 'f_measure': 0.6666666666666666}

    """
    # check validity of the input data
    assert len(labels) == len(outputs), "The number of labels and outputs should be the same"
    if not all([item.chagas is not None for item in outputs]):
        return {m: np.nan for m in ["challenge_score", "auroc", "auprc", "accuracy", "f_measure"]}
    assert all(
        [len(label["chagas"]) == len(output.chagas) for label, output in zip(labels, outputs)]
    ), "The number of 'chagas' labels and outputs should be the same"

    # convert the tensors to numpy arrays
    for label in labels:
        if isinstance(label["chagas"], torch.Tensor):
            label["chagas"] = label["chagas"].cpu().detach().numpy()
    # concatenate the labels and outputs
    labels = np.concat([label["chagas"] for label in labels])
    # probability_outputs is the probability of the positive class
    probability_outputs = np.concat([output.chagas_prob[:, 1] for output in outputs])
    binary_outputs = np.concat([output.chagas for output in outputs])
    # Evaluate the model outputs.
    challenge_score = compute_challenge_score(labels, probability_outputs)
    auroc, auprc = compute_auc(labels, probability_outputs)
    accuracy = compute_accuracy(labels, binary_outputs)
    f_measure = compute_f_measure(labels, binary_outputs)
    tpr = compute_chagas_tpr(labels, binary_outputs)

    return make_serializable(
        {
            "challenge_score": challenge_score,
            "auroc": auroc,
            "auprc": auprc,
            "accuracy": accuracy,
            "f_measure": f_measure,
            "tpr": tpr,
        }
    )


def compute_chagas_tpr(
    labels: np.ndarray,
    binary_outputs: np.ndarray,
) -> float:
    """Compute the TPR for the "chagas" prediction (binary classification) task.

    Parameters
    ----------
    labels : np.ndarray
        The labels for the records, of shape `(num_samples,)` with binary values.
    binary_outputs : np.ndarray
        The binary outputs for the records, of shape `(num_samples,)` with binary values.

    Returns
    -------
    float
        The computed TPR for the "chagas" prediction task.

    """
    A = compute_confusion_matrix(labels, binary_outputs)
    tp, fp, fn, tn = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
    tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan

    return tpr


def compute_challenge_score(labels, probability_outputs, max_fraction_positive=0.05):
    """Compute the challenge score for the "chagas" prediction (binary classification) task.

    The challenge score is defined as the true positive rate (TPR) for the model outputs,
    with the constraint that the number of positive model outputs is no more than 5% of the total instances.
    The TPR is defined as the number of true positive instances divided by the total number of positive instances.

    This function is copied from helper_code.py,
    and some printings are added to help debugging.

    Parameters
    ----------
    labels : np.ndarray
        The labels for the records, of shape `(num_samples,)` with binary values.
    probability_outputs : np.ndarray
        The probability outputs for the records, of shape `(num_samples,)` with probabilities (0 to 1).
    max_fraction_positive : float, default 0.05
        The maximum fraction of positive instances allowed.

    Returns
    -------
    float
        The computed challenge score for the "chagas" prediction task.

    """
    # Check the data.
    assert len(labels) == len(probability_outputs)
    num_instances = len(labels)
    max_num_positive_instances = int(max_fraction_positive * num_instances)

    # Convert the data to NumPy arrays, as needed, for easier indexing.
    labels = np.asarray(labels, dtype=np.float64)
    probability_outputs = np.asarray(probability_outputs, dtype=np.float64)

    # Collect the unique output values as the thresholds for the positive and negative classes.
    thresholds = np.unique(probability_outputs)
    thresholds = np.append(thresholds, thresholds[-1] + 1)
    thresholds = thresholds[::-1]
    num_thresholds = len(thresholds)

    idx = np.argsort(probability_outputs)[::-1]

    # Initialize the TPs, FPs, FNs, and TNs with no positive outputs.
    tp = np.zeros(num_thresholds)
    fp = np.zeros(num_thresholds)
    fn = np.zeros(num_thresholds)
    tn = np.zeros(num_thresholds)

    tp[0] = 0
    fp[0] = 0
    fn[0] = np.sum(labels == 1)
    tn[0] = np.sum(labels == 0)

    # Update the TPs, FPs, FNs, and TNs using the values at the previous threshold.
    i = 0
    for j in range(1, num_thresholds):
        tp[j] = tp[j - 1]
        fp[j] = fp[j - 1]
        fn[j] = fn[j - 1]
        tn[j] = tn[j - 1]

        while i < num_instances and probability_outputs[idx[i]] >= thresholds[j]:
            if labels[idx[i]] == 1:
                tp[j] += 1
                fn[j] -= 1
            else:
                fp[j] += 1
                tn[j] -= 1
            i += 1

    # Find the true positive rate so that the number of positive model outputs are no more than 5% of the total instances.
    k = num_thresholds
    for j in range(1, num_thresholds):
        if tp[j] + fp[j] > max_num_positive_instances:
            k = j - 1
            break

    print(f"Cutoff probability: {thresholds[k]}")

    if tp[k] + fn[k] > 0:
        tpr = tp[k] / (tp[k] + fn[k])
    else:
        tpr = float("nan")

    return tpr


def compute_arr_diag_metrics(
    labels: Sequence[Dict[str, Union[np.ndarray, torch.Tensor, List[dict]]]],
    outputs: Sequence[CINC2025Outputs],
) -> Dict[str, float]:
    """Compute the metrics for the arrhythmia diagnosis (multi-label classification) task.

    Parameters
    ----------
    labels : Sequence[Dict[str, Union[np.ndarray, torch.Tensor, List[dict]]]]
        The labels for the records.
    outputs : Sequence[CINC2025Outputs]
        The outputs for the records.

    Returns
    -------
    Dict[str, float]
        The computed challenge metrics for the arrhythmia diagnosis task.

    Examples
    --------
    >>> labels = [{"arr_diag": np.array([[0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0, 0, 0]])}]
    >>> outputs = [
    ...     CINC2025Outputs(
    ...         chagas=[0, 0],
    ...         chagas_prob=np.array([[0.9, 0.1], [0.7, 0.3]]),
    ...         arr_diag=[["RBBB", "AF"], ["SB", "ST"]],
    ...         arr_diag_prob=np.array([
    ...             [0.1, 0.9, 0.1, 0.1, 0.1, 0.7, 0.1, 0.1],
    ...             [0.3, 0.1, 0.1, 0.6, 0.7, 0.1, 0.1, 0.1],
    ...         ]),
    ...         arr_diag_classes=["1dAVb", "RBBB", "LBBB", "SB", "ST", "AF", "NORM", "OTHER"],
    ...     )
    ... ]
    >>> compute_arr_diag_metrics(labels, outputs)

    """
    # check validity of the input data
    assert len(labels) == len(outputs), "The number of labels and outputs should be the same"
    if not all([item.arr_diag is not None for item in outputs]):
        return {m: np.nan for m in ["f_measure", "accuracy", "sensitivity", "specificity", "precision"]}
    assert all(
        [len(label["arr_diag"]) == len(output.arr_diag) for label, output in zip(labels, outputs)]
    ), "The number of 'arr_diag' labels and outputs should be the same"

    # convert the tensors to numpy arrays
    for label in labels:
        if isinstance(label["arr_diag"], torch.Tensor):
            label["arr_diag"] = label["arr_diag"].cpu().detach().numpy()
    # concatenate the labels and outputs
    labels = np.concat([label["arr_diag"] for label in labels])
    probability_outputs = np.concat([output.arr_diag_prob for output in outputs], axis=0)
    # Evaluate the model outputs.
    cm = ClassificationMetrics(multi_label=True, macro=True)
    cm.compute(labels, probability_outputs, num_classes=len(outputs[0].arr_diag_classes))

    return make_serializable(
        {
            "f_measure": cm.f1_measure,
            "accuracy": cm.accuracy,
            "sensitivity": cm.sensitivity,
            "specificity": cm.specificity,
            "precision": cm.precision,
        }
    )
