"""
Miscellaneous functions.
"""

from functools import wraps
from numbers import Real
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml
from numpy._core.multiarray import normalize_axis_index
from numpy.lib._function_base_impl import array_function_dispatch

__all__ = [
    "func_indicator",
    "load_submission_log",
    "schmidt_spike_removal",
    "trim_zeros",
    "to_dtype",
    "remove_spikes_naive",
    "is_stdtypes",
]


def func_indicator(name: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            print("\n" + "-" * 100)
            print(f"  Start {name}  ".center(100, "-"))
            print("-" * 100 + "\n")
            func(*args, **kwargs)
            print("\n" + "-" * 100)
            print(f"  End {name}  ".center(100, "-"))
            print("-" * 100 + "\n")

        return wrapper

    return decorator


def load_submission_log() -> pd.DataFrame:
    """Load the submission log.

    Returns
    -------
    df_sub_log : pandas.DataFrame
        The submission log,
        sorted by challenge score in descending order.

    """
    path = Path(__file__).parents[1] / "submissions"
    df_sub_log = pd.DataFrame.from_dict(yaml.safe_load(path.read_text())["Official Phase"], orient="index").sort_values(
        "score", ascending=False
    )
    return df_sub_log


def schmidt_spike_removal(
    original_signal: np.ndarray,
    fs: int,
    window_size: float = 0.5,
    threshold: float = 3.0,
    eps: float = 1e-4,
) -> np.ndarray:
    """Spike removal using Schmidt algorithm.

    Parameters
    ----------
    original_signal : np.ndarray
        The original signal.
    fs : int
        The sampling frequency.
    window_size : float, default 0.5
        The sliding window size, with units in seconds.
    threshold : float, default 3.0
        The threshold (multiplier for the median value) for detecting spikes.
    eps : float, default 1e-4
        The epsilon for numerical stability.

    Returns
    -------
    despiked_signal : np.ndarray,
        The despiked signal.

    """
    window_size = round(fs * window_size)
    nframes, res = divmod(original_signal.shape[0], window_size)
    frames = original_signal[: window_size * nframes].reshape((nframes, window_size))
    if res > 0:
        nframes += 1
        frames = np.concatenate((frames, original_signal[-window_size:].reshape((1, window_size))), axis=0)
    MAAs = np.abs(frames).max(axis=1)  # of shape (nframes,)

    while len(np.where(MAAs > threshold * np.median(MAAs))[0]) > 0:
        frame_num = np.where(MAAs == MAAs.max())[0][0]
        spike_position = np.argmax(np.abs(frames[frame_num]))
        zero_crossings = np.where(np.diff(np.sign(frames[frame_num])))[0]
        spike_start = np.where(zero_crossings <= spike_position)[0]
        spike_start = zero_crossings[spike_start[-1]] if len(spike_start) > 0 else 0
        spike_end = np.where(zero_crossings >= spike_position)[0]
        spike_end = zero_crossings[spike_end[0]] + 1 if len(spike_end) > 0 else window_size
        frames[frame_num, spike_start:spike_end] = eps
        MAAs = np.abs(frames).max(axis=1)

    despiked_signal = original_signal.copy()
    if res > 0:
        despiked_signal[-window_size:] = frames[-1]
        nframes -= 1
    despiked_signal[: window_size * nframes] = frames[:nframes, ...].reshape((-1,))

    return despiked_signal


# Imported from numpy 2.2, for backward compatibility:
# https://github.com/numpy/numpy/blob/maintenance/2.2.x/numpy/lib/_function_base_impl.py#L1846-L1982

# Copyright (c) 2005-2024, NumPy Developers.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

#     * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.

#     * Redistributions in binary form must reproduce the above
#        copyright notice, this list of conditions and the following
#        disclaimer in the documentation and/or other materials provided
#        with the distribution.

#     * Neither the name of the NumPy Developers nor the names of any
#        contributors may be used to endorse or promote products derived
#        from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


def _arg_trim_zeros(filt):
    """Return indices of the first and last non-zero element.

    Parameters
    ----------
    filt : array_like
        Input array.

    Returns
    -------
    start, stop : ndarray
        Two arrays containing the indices of the first and last non-zero
        element in each dimension.

    See also
    --------
    trim_zeros

    Examples
    --------
    >>> import numpy as np
    >>> _arg_trim_zeros(np.array([0, 0, 1, 1, 0]))
    (array([2]), array([3]))
    """
    nonzero = (
        np.argwhere(filt)
        if filt.dtype != np.object_
        # Historically, `trim_zeros` treats `None` in an object array
        # as non-zero while argwhere doesn't, account for that
        else np.argwhere(filt != 0)
    )
    if nonzero.size == 0:
        start = stop = np.array([], dtype=np.intp)
    else:
        start = nonzero.min(axis=0)
        stop = nonzero.max(axis=0)
    return start, stop


def _trim_zeros(filt, trim=None, axis=None):
    return (filt,)


@array_function_dispatch(_trim_zeros)
def trim_zeros(filt, trim="fb", axis=None):
    """Remove values along a dimension which are zero along all other.

    Parameters
    ----------
    filt : array_like
        Input array.
    trim : {"fb", "f", "b"}, optional
        A string with 'f' representing trim from front and 'b' to trim from
        back. By default, zeros are trimmed on both sides.
        Front and back refer to the edges of a dimension, with "front" refering
        to the side with the lowest index 0, and "back" refering to the highest
        index (or index -1).
    axis : int or sequence, optional
        If None, `filt` is cropped such, that the smallest bounding box is
        returned that still contains all values which are not zero.
        If an axis is specified, `filt` will be sliced in that dimension only
        on the sides specified by `trim`. The remaining area will be the
        smallest that still contains all values wich are not zero.

    Returns
    -------
    trimmed : ndarray or sequence
        The result of trimming the input. The number of dimensions and the
        input data type are preserved.

    Notes
    -----
    For all-zero arrays, the first axis is trimmed first.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array((0, 0, 0, 1, 2, 3, 0, 2, 1, 0))
    >>> np.trim_zeros(a)
    array([1, 2, 3, 0, 2, 1])

    >>> np.trim_zeros(a, trim='b')
    array([0, 0, 0, ..., 0, 2, 1])

    Multiple dimensions are supported.

    >>> b = np.array([[0, 0, 2, 3, 0, 0],
    ...               [0, 1, 0, 3, 0, 0],
    ...               [0, 0, 0, 0, 0, 0]])
    >>> np.trim_zeros(b)
    array([[0, 2, 3],
           [1, 0, 3]])

    >>> np.trim_zeros(b, axis=-1)
    array([[0, 2, 3],
           [1, 0, 3],
           [0, 0, 0]])

    The input data type is preserved, list/tuple in means list/tuple out.

    >>> np.trim_zeros([0, 1, 2, 0])
    [1, 2]

    """
    filt_ = np.asarray(filt)

    trim = trim.lower()
    if trim not in {"fb", "bf", "f", "b"}:
        raise ValueError(f"unexpected character(s) in `trim`: {trim!r}")

    start, stop = _arg_trim_zeros(filt_)
    stop += 1  # Adjust for slicing

    if start.size == 0:
        # filt is all-zero -> assign same values to start and stop so that
        # resulting slice will be empty
        start = stop = np.zeros(filt_.ndim, dtype=np.intp)
    else:
        if "f" not in trim:
            start = (None,) * filt_.ndim
        if "b" not in trim:
            stop = (None,) * filt_.ndim

    if len(start) == 1:
        # filt is 1D -> don't use multi-dimensional slicing to preserve
        # non-array input types
        sl = slice(start[0], stop[0])
    elif axis is None:
        # trim all axes
        sl = tuple(slice(*x) for x in zip(start, stop))
    else:
        # only trim single axis
        axis = normalize_axis_index(axis, filt_.ndim)
        sl = (slice(None),) * axis + (slice(start[axis], stop[axis]),) + (...,)

    trimmed = filt[sl]
    return trimmed


def to_dtype(data: np.ndarray, dtype: np.dtype = np.float32) -> np.ndarray:
    """Convert the data to the specified data type,
    and fill the missing values with zeros.

    Parameters
    ----------
    data : np.ndarray
        The input data.
    dtype : np.dtype, default np.float32
        The expected data type.

    Returns
    -------
    data : np.ndarray
        The converted data.

    """
    if data.dtype == dtype:
        return data
    if data.dtype in (np.int8, np.uint8, np.int16, np.int32, np.int64):
        data = data.astype(dtype) / (np.iinfo(data.dtype).max + 1)
    else:
        data = data.astype(dtype)
    data[~np.isfinite(data)] = 0.0

    return data


def remove_spikes_naive(sig: np.ndarray, threshold: Real = 20, inplace: bool = True) -> np.ndarray:
    """Remove signal spikes using a naive method.

    This is a method proposed in entry 0416 of CPSC2019.
    `spikes` here refers to abrupt large bumps with (abs) value
    larger than the given threshold,
    or nan values (read by `wfdb`).
    Do **NOT** confuse with `spikes` in paced rhythm.

    Parameters
    ----------
    sig : numpy.ndarray
        1D or 2D signal with potential spikes.
        If is 2D, it should be of lead-first format.
    threshold : numbers.Real, optional
        Values of `sig` that are larger than `threshold` will be removed.
    inplace : bool, optional
        Whether to modify `sig` in place or not.

    Returns
    -------
    numpy.ndarray
        Signal with `spikes` removed.

    Examples
    --------
    .. code-block:: python

        sig = np.random.randn(1000)
        pos = np.random.randint(0, 1000, 10)
        sig[pos] = 100
        sig = remove_spikes_naive(sig)
        pos = np.random.randint(0, 1000, 1)
        sig[pos] = np.nan
        sig = remove_spikes_naive(sig)

    """
    assert sig.ndim <= 2, f"Only 1D or 2D signal is supported, but got {sig.ndim}D signal"
    if sig.ndim == 2:
        return np.apply_along_axis(lambda x: remove_spikes_naive(x, threshold, inplace), axis=0, arr=sig)
    dtype = sig.dtype
    b = list(
        filter(
            lambda k: k > 0,
            np.argwhere(np.logical_or(np.abs(sig) > threshold, np.isnan(sig))).squeeze(-1),
        )
    )
    if not inplace:
        sig = sig.copy()
    if abs(sig[0]) > threshold or np.isnan(sig[0]):
        sig[0] = 0
    for k in b:
        sig[k] = sig[k - 1]
    return sig.astype(dtype)


def is_stdtypes(x: Any) -> bool:
    """Check if an object is an instance of Python
    built-in standard data types.

    For more information, see [#stdtypes]_.

    Parameters
    ----------
    x : Any
        Object to be checked.

    Returns
    -------
    bool
        True if `x` is a standard data type, False otherwise.

    References
    ----------
    .. [#stdtypes] https://docs.python.org/3/library/stdtypes.html

    """
    # return type(x).__module__ == "builtins"
    # fmt: off
    builtins = (
        int, float, complex,
        bool,
        list, tuple, range,
        str,
        bytes, bytearray, memoryview,
        set, frozenset,
        dict,
        type(None),
    )
    # fmt: on
    return isinstance(x, builtins)
