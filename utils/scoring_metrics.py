from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch_ecg.components.metrics import ClassificationMetrics
from torch_ecg.utils.misc import make_serializable

from helper_code import compute_accuracy, compute_auc, compute_f_measure
from outputs import CINC2025Outputs

__all__ = [
    "compute_challenge_metrics",
]


def compute_challenge_metrics(
    labels: Sequence[Dict[str, Union[np.ndarray, torch.Tensor, List[dict]]]],
    outputs: Sequence[CINC2025Outputs],
    keeps: Optional[Union[str, Sequence[str]]] = None,
    verbose: bool = False,
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
    verbose : bool, default False
        Whether to print some debug information.

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
    fraction_capacity: float = 0.05,
    verbose: bool = False,
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
    fraction_capacity : float, default 0.05
        The maximum fraction of positive instances allowed for the challenge score.
    verbose : bool, default False
        Whether to print some debug information.

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
    challenge_score = compute_challenge_score(labels, probability_outputs, fraction_capacity=fraction_capacity, verbose=verbose)
    auroc, auprc = compute_auc(labels, probability_outputs)
    accuracy = compute_accuracy(labels, binary_outputs)
    f_measure = compute_f_measure(labels, binary_outputs)
    tpr = compute_chagas_tpr(labels, binary_outputs, verbose)

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


def compute_confusion_matrix(labels: np.ndarray, outputs: np.ndarray) -> np.ndarray:
    """Compute the confusion matrix for binary classification.

    Parameters
    ----------
    labels : np.ndarray
        The ground truth labels, of shape `(num_samples,)` with binary values.
    outputs : np.ndarray
        The predicted outputs, of shape `(num_samples,)` with binary values.

    Returns
    -------
    np.ndarray
        The confusion matrix, of shape `(2, 2)`, with the following structure:
        [[TP, FN],
         [FP, TN]]

    """
    assert np.shape(labels) == np.shape(outputs)
    num_instances = len(labels)

    A = np.zeros((2, 2))
    for i in range(num_instances):
        if labels[i] == 1 and outputs[i] == 1:
            A[0, 0] += 1
        elif labels[i] == 1 and outputs[i] == 0:
            A[0, 1] += 1
        elif labels[i] == 0 and outputs[i] == 1:
            A[1, 0] += 1
        elif labels[i] == 0 and outputs[i] == 0:
            A[1, 1] += 1
        else:
            raise ValueError(f"{labels[i]} and/or {outputs[i]} not valid.")

    return A


def compute_chagas_tpr(
    labels: np.ndarray,
    binary_outputs: np.ndarray,
    verbose: bool = False,
) -> float:
    """Compute the TPR for the "chagas" prediction (binary classification) task.

    Parameters
    ----------
    labels : np.ndarray
        The labels for the records, of shape `(num_samples,)` with binary values.
    binary_outputs : np.ndarray
        The binary outputs for the records, of shape `(num_samples,)` with binary values.
    verbose : bool, default False
        Whether to print the confusion matrix values.
        If True, the TP, FN, TN, and FP values will be printed.

    Returns
    -------
    float
        The computed TPR for the "chagas" prediction task.

    """
    A = compute_confusion_matrix(labels, binary_outputs)
    tp, fn, fp, tn = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
    tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    if verbose:
        print(f"compute_chagas_tpr TP: {tp}")
        print(f"compute_chagas_tpr FN: {fn}")
        print(f"compute_chagas_tpr TN: {tn}")
        print(f"compute_chagas_tpr FP: {fp}")

    return tpr


def compute_challenge_score(
    labels, probability_outputs, fraction_capacity=0.05, num_permutations=10**4, seed=12345, verbose=False
):
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
    fraction_capacity : float, default 0.05
        The maximum fraction of positive instances allowed.
    num_permutations: int, default 10**4
        Number of permutations for numerical stability.
    seed: int, default 12345
        Random seed for permutations.
    verbose : bool, default False
        Whether to print the confusion matrix values.
        If True, the TP, FN, TN, and FP values will be printed.

    Returns
    -------
    float
        The computed challenge score for the "chagas" prediction task.

    """
    # Check the data.
    assert len(labels) == len(probability_outputs)
    num_instances = len(labels)
    capacity = int(fraction_capacity * num_instances)

    # Convert the data to NumPy arrays, as needed, for easier indexing.
    labels = np.asarray(labels, dtype=np.float64)
    probability_outputs = np.asarray(probability_outputs, dtype=np.float64)

    # Permute the labels and outputs so that we can approximate the expected confusion matrix for "tied" probabilities.
    tp = np.zeros(num_permutations)
    fp = np.zeros(num_permutations)
    fn = np.zeros(num_permutations)
    tn = np.zeros(num_permutations)

    if seed is not None:
        np.random.seed(seed)

    for i in range(num_permutations):
        permuted_idx = np.random.permutation(np.arange(num_instances))
        permuted_labels = labels[permuted_idx]
        permuted_outputs = probability_outputs[permuted_idx]

        ordered_idx = np.argsort(permuted_outputs, stable=True)[::-1]
        ordered_labels = permuted_labels[ordered_idx]

        tp[i] = np.sum(ordered_labels[:capacity] == 1)
        fp[i] = np.sum(ordered_labels[:capacity] == 0)
        fn[i] = np.sum(ordered_labels[capacity:] == 1)
        tn[i] = np.sum(ordered_labels[capacity:] == 0)

    tp = np.mean(tp)
    fp = np.mean(fp)
    fn = np.mean(fn)
    tn = np.mean(tn)

    # Compute the true positive rate.
    if tp + fn > 0:
        tpr = tp / (tp + fn)
    else:
        tpr = float("nan")

    # print(f"compute_challenge_score cutoff probability: {thresholds[k]}")
    if verbose:
        print(f"compute_challenge_score TP: {tp}")
        print(f"compute_challenge_score FN: {fn}")
        print(f"compute_challenge_score TN: {tn}")
        print(f"compute_challenge_score FP: {fp}")

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
