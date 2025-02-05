from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from torch_ecg.utils.misc import list_sum

from helper_code import compute_accuracy, compute_auc, compute_challenge_score, compute_f_measure
from outputs import CINC2025Outputs

__all__ = [
    "compute_challenge_metrics",
]


def compute_challenge_metrics(
    labels: Sequence[Dict[str, Union[np.ndarray, List[dict]]]],
    outputs: Sequence[CINC2025Outputs],
    keeps: Optional[Union[str, Sequence[str]]] = None,
) -> Dict[str, float]:
    """Compute the challenge metrics.

    Parameters
    ----------
    labels : Sequence[Dict[str, Union[np.ndarray, List[dict]]]]
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

    """
    metrics = {}
    if keeps is None:
        keeps = ["chagas", "arr_diag"]
    elif isinstance(keeps, str):
        keeps = [keeps]
    keeps = [keep.lower() for keep in keeps]
    if "chagas" in keeps:
        metrics.update({f"chagas_{metric}": value for metric, value in compute_chagas_metrics(labels, outputs).items()})
    if "arr_diag" in keeps:
        metrics.update({f"arr_diag_{metric}": value for metric, value in compute_arr_diag_metrics(labels, outputs).items()})
    return metrics


def compute_chagas_metrics(
    labels: Sequence[Dict[str, Union[np.ndarray, List[dict]]]],
    outputs: Sequence[CINC2025Outputs],
) -> Dict[str, float]:
    """Compute the metrics for the "chagas" prediction (binary classification) task.

    Parameters
    ----------
    labels : Sequence[Dict[str, Union[np.ndarray, List[dict]]]]
        The labels for the records, containing the "chagas" field.
        The "chagas" field is a 1D numpy array of shape `(num_samples,)` with binary values,
        or a 2D numpy array of shape `(num_samples, num_classes)` with probabilities (0 or 1).
    outputs : Sequence[CINC2025Outputs]
        The outputs for the records, containing the "chagas" field.
        The "chagas" field is a 1D numpy array of shape `(num_samples,)` with binary values,
        or a 2D numpy array of shape `(num_samples, num_samples)` with probabilities (0 to 1).

    Returns
    -------
    Dict[str, float]
        The computed challenge metrics for the "chagas" prediction task.

    """
    assert len(labels) == len(outputs), "The number of labels and outputs should be the same"
    if not all([item.chagas is not None for item in outputs]):
        return {"f_measure": np.nan}
    assert all(
        [len(label["chagas"]) == len(output.chagas) for label, output in zip(labels, outputs)]
    ), "The number of 'chagas' labels and outputs should be the same"
    # concatenate the labels and outputs
    # labels = np.concatenate([label["chagas"] for label in labels], axis=0)
    # probability_outputs = np.concatenate([output.chagas for output in outputs], axis=0)
    labels = list_sum([label["chagas"] for label in labels])
    # probability_outputs is the probability of the positive class
    probability_outputs = list_sum([output.chagas_prob[:, 1] for output in outputs])
    binary_outputs = list_sum([output.chagas for output in outputs])
    # Evaluate the model outputs.
    challenge_score = compute_challenge_score(labels, probability_outputs)
    auroc, auprc = compute_auc(labels, probability_outputs)
    accuracy = compute_accuracy(labels, binary_outputs)
    f_measure = compute_f_measure(labels, binary_outputs)

    return {
        "challenge_score": challenge_score,
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": accuracy,
        "f_measure": f_measure,
    }


def compute_arr_diag_metrics(
    labels: Sequence[Dict[str, Union[np.ndarray, List[dict]]]],
    outputs: Sequence[CINC2025Outputs],
) -> Dict[str, float]:
    """Compute the metrics for the arrhythmia diagnosis (multi-label classification) task.

    Parameters
    ----------
    labels : Sequence[Dict[str, Union[np.ndarray, List[dict]]]]
        The labels for the records.
    outputs : Sequence[CINC2025Outputs]
        The outputs for the records.

    Returns
    -------
    Dict[str, float]
        The computed challenge metrics for the arrhythmia diagnosis task.

    """
    raise NotImplementedError
    # assert len(labels) == len(outputs), "The number of labels and outputs should be the same"
